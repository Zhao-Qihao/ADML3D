# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset
import os.path as osp


@DATASETS.register_module()
class CustomDataset(Det3DDataset):
    r"""Custom Dataset.

    This class serves as the API for experiments on the custom Dataset.


    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """

    METAINFO = {
        'classes': ('car','truck','bus','bicycle','pedestrian','traffic_cone','barrier'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42),(180, 135, 0), (0, 60, 100), (0, 0, 142), (0, 0, 70)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = False,
                 test_mode: bool = False,
                 **kwargs):
        assert box_type_3d.lower() in ['lidar']
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of 3D ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            # empty instance
            anns_results = dict()
            anns_results['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            anns_results['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
            return anns_results
        gt_bboxes_3d = ann_info['gt_bboxes_3d']
        gt_labels_3d = ann_info['gt_labels_3d']

        # 检查是否缺少速度信息（vx, vy）
        if gt_bboxes_3d.shape[1] == 7:  # 原始数据只有7维（无速度）
            # 补零到9维：添加速度（vx=0, vy=0）
            vxvy = np.zeros((len(gt_bboxes_3d), 2), dtype=gt_bboxes_3d.dtype)
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, vxvy], axis=1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        return anns_results
