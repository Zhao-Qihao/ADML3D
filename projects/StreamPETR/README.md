# StreamPETR
## Results and models

### 1. Setup

- Run setup script

```sh
cd projects/StreamPETR && pip install -e .
```
use ```pip show nuscenes-devkit``` to find and comment the blow code
```
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."
```
### 2. Train

- Run training on nuscenes dataset with appropriate configs

```sh
# Single GPU training
python tools/train.py projects/StreamPETR/configs/nuscenes/nuscenes_vov_flash_320x800_baseline.py
```

```sh
# Multi GPU training

bash tools/dist_train.sh projects/StreamPETR/configs/nuscenes/nuscenes_vov_flash_320x800_baseline.py 4
```

### 3. Evaluation

- Run evaluation on a test set, please select experiment config accordingly

```sh
# Evaluation for nuscenes
python tools/test.py projects/StreamPETR/configs/nuscenes/nuscenes_vov_flash_320x800_baseline.py work_dirs/nuscenes_vov_flash_320x800_baseline/epoch_35.pth
```

### 4. Visualization

- Run inference and visualize bounding boxes.

```sh
# Inference for nuscenes
python tools/visualize/visualize_bboxes_cameraonly.py projects/StreamPETR/configs/nuscenes/nuscenes_vov_flash_320x800_baseline.py.py work_dirs/nuscenes_vov_flash_320x800_baseline.py/epoch_35.pth
```
### 5. Deploy

- Make onnx files for a StreamPETR model

```sh
CONFIG_PATH=/path/to/config
CHECKPOINT_PATH=/path/to/checkpoint
python3 projects/StreamPETR/deploy/torch2onnx.py $CONFIG_PATH --section extract_img_feat --checkpoint $CHECKPOINT_PATH
python3 projects/StreamPETR/deploy/torch2onnx.py $CONFIG_PATH --section pts_head_memory --checkpoint $CHECKPOINT_PATH
python3 projects/StreamPETR/deploy/torch2onnx.py $CONFIG_PATH --section position_embedding --checkpoint $CHECKPOINT_PATH
```

## Reference

- [StreamPETR Official](https://github.com/exiawsh/StreamPETR/tree/main)
- [NVIDIDA DL4AGX TensorRT](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/streampetr-trt)
