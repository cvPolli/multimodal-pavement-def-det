from models.yolov8_imu_multimodal import MultimodalYOLO

if __name__ == "__main__":
    multimodal_yolo = MultimodalYOLO(
        pretrained_yolo_weights="ultralytics/cfg/models/v8/yolov8-imu.yaml"
    )
    multimodal_yolo.train(
        data="pavment_defects_standard.yaml",
        imgsz=416,
        epochs=1,
        batch=16,
        name="test_multimodal",
        exist_ok=True,
    )
