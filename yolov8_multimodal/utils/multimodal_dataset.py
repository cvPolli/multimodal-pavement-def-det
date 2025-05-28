import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MultimodalDataset(Dataset):
    def __init__(self, dataset_folder: str, transform=None):
        self.images_folder = dataset_folder
        self.annotations_folder = dataset_folder
        self.tabular_folder = dataset_folder
        self.transform = transform

        self.image_files = sorted(
            [f for f in os.listdir(dataset_folder) if f.endswith(".jpg")]
        )
        self.annotation_files = sorted(
            [f for f in os.listdir(dataset_folder) if f.endswith(".txt")]
        )
        self.tabular_files = sorted(
            [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
        )

        # Ensure that the files match up
        assert (
            len(self.image_files)
            == len(self.annotation_files)
            == len(self.tabular_files)
        ), "Mismatch between image, annotation, and tabular files"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_folder, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Load YOLO annotations
        annotation_path = os.path.join(
            self.annotations_folder, self.annotation_files[idx]
        )
        with open(annotation_path, "r") as file:
            yolo_annotations = file.readlines()

        # Process annotations
        annotations = []
        for line in yolo_annotations:
            parts = line.strip().split()
            label = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            annotations.append([label, x_center, y_center, width, height])

        annotations = torch.tensor(annotations)

        # Load tabular data
        tabular_path = os.path.join(self.tabular_folder, self.tabular_files[idx])
        tabular_data = pd.read_csv(
            tabular_path
        ).values.flatten()  # Assuming each CSV has a single row
        tabular_data_category = tabular_data[-1]
        tabular_data_features = tabular_data[:-1]
        tabular_data = torch.tensor(tabular_data_features, dtype=torch.float32)

        return image, tabular_data, annotations, tabular_data_category
