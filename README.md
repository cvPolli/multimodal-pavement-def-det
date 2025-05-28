# Detection of Pavement Defects on Roads Using a Multimodal YOLOv8 with Image and IMU Data
## Authors
[Arthur Jose Antocevicz Polli](https://github.com/cvPolli/)

[Ricardo Dutra da Silva](ricardodutr@gmail.com)

[Rodrigo Minetto](https://github.com/rminetto)

## Datasets and Weights
The dataset, which includes extracted ROI images, image annotations in YOLO format, and corresponding IMU data, is available on [Google Drive](https://drive.google.com/file/d/1EUDaigDOiuvBKGXffoAy0i3YdjaSURci/view?usp=sharing). Additionally, the video containing IMU data and other GoPro payloads can be accessed [here](https://drive.google.com/drive/folders/1t4CtfLE8O3-m3UhOX5kRh7ehYhGnBdgT?usp=sharing). For more information on how to extract these data from the video, refer to the official [GoPro GPMF parser repository](https://github.com/gopro/gpmf-parser).

The multimodal weights can be found in [runs/detect/yolov8-imu-multimodal]() along with all other information related to the YOLO experiments.

The MLP weights are available in the [weights]() directory.


# License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
