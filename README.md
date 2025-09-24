This repo show how to train 3D object detection algorithms on custom dataset or Nuscenes dataset based on mmdetection3d. Include: creating a custom dataset, training and evaluating on custom dataset metrics.
# Getting Started
```
git clone https://github.com/Zhao-Qihao/mmdetection3d.git
```

## Installation
Refer to [official mmdetection3d repo installation](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html) to prepare the environment.

## Supported
Here are the new features of this repository compared to MMDetection3D. 
* [BEVFusion](./projects/BEVFusion/)
  * Training with custom data
  * Replace the SparseConvolution in original BEVFusion with traveller59 version
  * Export checkpoint to onnx format
* [StreamPETR](./projects/StreamPETR/)
  * Update to MMDetection3D v1.4.0 and torch 2.x
  * Export checkpoint to onnx format
## Custom Dataset Preparation
Prepare the dataset following [prepare_data](https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/customize_dataset.html)
The directory should be organized as follows. Refer to [xbzl_data](https://github.com/Zhao-Qihao/xbzl_data).
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── scene_1
│   │   ├── scene_2
│   │   ├── scene_3
│   │   ├── scene_4
│   │   ├── trainval.yaml
```

## Create custom data info
```python
python tools/create_data.py custom --root-path ./data/custom --out-dir ./data/custom --extra-tag custom
```
notice that there are 7class in my custom dataset, ['car','truck','bus','bicycle','pedestrian','traffic_cone','barrier'], you can change the class name in `mmdet3d/datasets/custom_dataset.py` to fit your dataset.



