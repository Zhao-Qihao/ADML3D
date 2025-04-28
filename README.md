This repo show how to train BEVFusion on custom dataset. Include: creating a custom dataset, training and evaluating on custom dataset metrics.
# Getting Started
```
git clone https://github.com/Zhao-Qihao/mmdetection3d.git
git checkout customdata
```
## Installation
Refer to [installation](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html) to prepare the environment.
## Dataset Preparation
Prepare the dataset following [prepare_data](https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/customize_dataset.html)
The directory should be organized as follows
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── calibs
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── images
│   │   │   ├── images_0
│   │   │   │   ├── 000000.png
│   │   │   │   ├── 000001.png
│   │   │   │   ├── ...
│   │   │   ├── images_1
│   │   │   ├── images_2
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

## Create custom data info
```python
python tools/create_data.py custom --root-path ./data/custom --out-dir ./data/custom --extra-tag custom
```
notice that there are 7class in my custom dataset, ['car','truck','bus','bicycle','pedestrian','traffic_cone','barrier'], you can change the class name in `mmdet3d/datasets/custom_dataset.py` to fit your dataset.

## Training
for lidar only training
```python
bash tools/dist_train.sh projects/BEVFusion/configs/lidar_custom.py 4
```
for lidar-cam training
```python
bash tools/dist_train.sh projects/BEVFusion/configs/lidar-cam_custom.py 4
```
notice that you should change the `data_prefix` and `point_load_dim` in `projects/BEVFusion/configs/lidar_custom.py` to your own. 
