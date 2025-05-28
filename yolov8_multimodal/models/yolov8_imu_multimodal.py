from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.results import Results
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, OBB, SPP, SPPELAN, SPPF, ADown, Bottleneck, BottleneckCSP,
                                    C2f, C2fAttn, C3Ghost, C3x, CBFuse, CBLinear, Classify, Concat, Conv, Conv2,
                                    ConvTranspose, Detect, DetectImu, DWConv, DWConvTranspose2d, Focus, GhostBottleneck,
                                    GhostConv, HGBlock, HGStem, ImagePoolingAttn, Pose, RepC3, RepConv, RepNCSPELAN4,
                                    ResNetLayer, RTDETRDecoder, Segment, Silence, WorldDetect)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, ops, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8OBBLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts,
                                           make_divisible, model_info, scale_img, time_sync)
from yolov8_multimodal.utils.config_utils import yaml_load
from yolov8_multimodal.utils.multimodal_dataset import MultimodalDataset

try:
    import thop
except ImportError:
    thop = None

# preds = (
#     self.forward(batch["img"], batch["imu_features"])
#     if batch.get("imu_features")
#     else self.forward(batch["img"]) if preds is None else preds
# )


def custom_collate_fn(batch):
    images, tabular_data, annotations, labels = zip(*batch)

    images = torch.stack(images)
    tabular_data = torch.stack(tabular_data)
    labels = torch.tensor(labels)

    # Handling variable number of annotations
    max_num_annotations = max(ann.shape[0] for ann in annotations)
    padded_annotations = torch.zeros((len(annotations), max_num_annotations, 5))

    for i, ann in enumerate(annotations):
        if ann.shape[0] > 0:
            padded_annotations[i, : ann.shape[0], :] = ann

    return images, tabular_data, padded_annotations, labels


class ImuClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImuClassificationModel, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)

        self.dropout = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = nn.ReLU()(x)

        x = self.dropout(x)
        features = x  # Extracted feature layer
        x = self.fc5(x)
        return x, features  # Return both the final output and the features


class MultimodalYOLOModel(nn.Module):
    def __init__(self, pretrained_model: DetectionModel, imu_feature_input_size: int):
        super(MultimodalYOLOModel, self).__init__()
        self.model = pretrained_model
        self.imu_classification_model = ImuClassificationModel(
            input_size=imu_feature_input_size, num_classes=2
        )
        self.names = None
        self.args = get_cfg()

    def _add_custom_head(self, imu_feature_input_size: int):
        # Add your custom layers here
        # self.model.add_module(
        #     "custom_head",
        #     torch.nn.Sequential(
        #         torch.nn.Conv2d(1024, 512, 3, stride=1, padding=1),  # Example new layer
        #         torch.nn.ReLU(),
        #     ),
        # )

        self.model.add_module(
            "imu_features_classification",
            ImuClassificationModel(
                input_size=imu_feature_input_size, num_classes=self.model.nc
            ),
        )
        print(self.model)

    def forward(self, yolo_input, imu_features, *args, **kwargs):
        # super(ImuClassificationModel, self).__init__()
        # print(self.model)
        # return super().forward(x, *args, **kwargs)
        if isinstance(yolo_input, dict) and isinstance(
            imu_features, dict
        ):  # for cases of training and validating while training.
            return self.loss(yolo_input, imu_features, *args, **kwargs)
        return self.predict(yolo_input, imu_features, *args, **kwargs)

    def predict(
        self,
        yolo_input,
        imu_features,
        profile=False,
        visualize=True,
        augment=False,
        embed=None,
    ):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        # if augment:
        #     return self._predict_augment(yolo_input, imu_features)
        return self._predict_once(yolo_input, imu_features, profile, visualize, embed)

    def _predict_once(self, x, imu_features, profile=False, visualize=True, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            if isinstance(m, DetectImu):
                # TODO: adjust DetectImu to handle imu_features
                classification, imu_features = self.imu_classification_model(imu_features)
                m.training = False
                x = m(x, imu_features)
                print(classification.shape)
                print(imu_features.shape)
                # if visualize:
                #     for index, _x in enumerate(x):
                #         feature_visualization(
                #             _x,
                #             m.type,
                #             f"{m.i}_{index}",
                #             n=1000,
                #             save_dir=Path("./yolo_features"),
                #         )
            else:
                x = m(x)  # run
            y.append(x if m.i in self.model.save else None)  # save output

            if embed and m.i in embed:
                embeddings.append(
                    nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                )  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(
            x, list
        )  # is final layer list, copy input as inplace fix
        flops = (
            thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2
            if thop
            else 0
        )  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self.model)

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)


class MultimodalYOLO:
    def __init__(
        self, pretrained_yolo_weights: str = "yolov8m.pt", tabular_input_size: int = 300
    ) -> None:
        self.model = self._get_pretrained_yolo(
            pretrained_yolo_weights, tabular_input_size
        )
        self.fp16 = False
        self.num_classes = 7

    def _get_pretrained_yolo(
        self, pretrained_yolo_weights: str, input_size: int
    ) -> MultimodalYOLOModel:
        # Load the pre-trained YOLO model
        pretrained_model = YOLO(pretrained_yolo_weights)
        # Instantiate models
        multimodal_yolo = MultimodalYOLOModel(pretrained_model.model, input_size)
        return multimodal_yolo

    def _get_train_dataloader(self, data):
        # Assuming `images`, `tabular_data`, and `labels` are numpy arrays or similar
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = MultimodalDataset(
            dataset_folder=f"{data['path']}/{data['train']}", transform=transform
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn
        )
        return train_dataloader

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose(
                (0, 3, 1, 2)
            )  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        # im = im.to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            0.25,
            0.7,
            agnostic=False,
            max_det=300,
            classes=None,
            nc=self.num_classes,
        )

        if not isinstance(
            orig_imgs, list
        ):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # img_path = self.batch[0][i]
            img_path = None
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
            )
        return results

    def train(
        self, data: str, imgsz: int, epochs: int, batch: int, name: str, exist_ok: bool
    ):
        data = yaml_load(data)

        train_dataloader = self._get_train_dataloader(data)
        self.model = self.model.cuda()

        # criterion = nn.CrossEntropyLoss()  # Or other appropriate loss for your task
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.names = data["names"]
        print(self.model.parameters())
        for epoch in range(epochs):
            # self.model.eval()
            running_loss = 0.0

            for images0, tabular_data, labels, tabular_labels in train_dataloader:
                images0 = images0.cuda()
                tabular_data = tabular_data.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                images = self.preprocess(images0)
                # Forward pass
                outputs = self.model(images, tabular_data)

                self.postprocess(outputs, images, images0)
                # Calculate loss
                loss = self.model.loss(labels, outputs)
                # loss = criterion(outputs, labels)

                # Calculate loss
                # loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_dataloader):.4f}"
            )
