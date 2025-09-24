## Introduction

We implement BEVFusion and support training and testing on NuScenes dataset and export to onnx.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Compiling operations on CUDA

**Note** that the voxelization OP in the original implementation of `BEVFusion` is different from the implementation in MMCV. If you want to use the original pretrained model [here](https://github.com/mit-han-lab/bevfusion/blob/main/README.md), you need to use the original implementation of voxelization OP.

```python
python projects/BEVFusion/setup.py develop
```

### Demo

Run a demo on NuScenes data using [BEVFusion model](https://drive.google.com/file/d/1QkvbYDk4G2d6SZoeJqish13qSyXA4lp3/view?usp=share_link):

```shell
python projects/BEVFusion/demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${CHECKPOINT_FILE} --cam-type all --score-thr 0.2 --show
```

### Training commands
We support two dataset type, nuscenes dataset and custom dataset.
Below is the training, testing, and exporting method on the nuScenes dataset. Custom datasets are similar. Your custom dataset format should refer to [xbzl_data
](https://github.com/Zhao-Qihao/xbzl_data/tree/master), and notice that you should change the `data_prefix` and `point_load_dim` in [lidar_custom.py](projects/BEVFusion/configs/lidar_custom.py) to your own. 
1. You should train the lidar-only detector first:

```bash
bash tools/dist_train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 8
```

2. Download the [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth). Given the image pre-trained backbone and the lidar-only pre-trained detector, you could train the lidar-camera fusion model:

```bash
bash tools/dist_train.sh projects/BEVFusion/configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 8 --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}
```

**Note** that if you want to reduce CUDA memory usage and computational overhead, you could directly add `--amp` on the tail of the above commands. The model under this setting will be trained in fp16 mode.

### Testing commands

In MMDetection3D's root directory, run the following command to test the model:

```bash
bash tools/dist_test.sh projects/BEVFusion/configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${CHECKPOINT_PATH} 8
```

### Deployment
#### Sparse convolutions support
Sparse convolutions are not deployable by default. In the deployment we follow the instructions found in the [SparseConvolution](../SparseConvolution/README.md) project to enable this feature.

Note: we only support traveller59's backend during deployment, but the model checkpoints can correspond to either backend.

#### ONNX Export


To export an ONNX, use the following command:

```bash
DEPLOY_CFG=projects/BEVFusion/configs/deploy/bevfusion_camera_lidar_tensorrt_dynamic.py
MODEL_CFG=...
CHECKPOINT_PATH=...
WORK_DIR=...

python projects/BEVFusion/deploy/export.py \
  ${DEPLOY_CFG} \
  ${MODEL_CFG} \
  ${CHECKPOINT_PATH} \
  --device cuda:0 \
  --work-dir ${WORK_DIR}
```