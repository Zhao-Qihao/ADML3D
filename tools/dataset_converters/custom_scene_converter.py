import os
from os import path as osp
import mmengine
import numpy as np
import json
import yaml  # 用于读取 YAML 文件

class_names = ['car', 'truck', 'bus', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier']
class_order = [0, 1, 2, 3, 4, 5, 6]
categories = dict(zip(class_names, class_order))

def create_custom_dataset_infos(root_path, info_prefix):
    train_infos, val_infos = _fill_trainval_infos(root_path)
    metainfo = dict(categories=categories,
                    dataset="custom",
                    version="v1.0")

    if train_infos:
        data = dict(data_list=train_infos, metainfo=metainfo)
        info_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        print(f"Saving training info to {info_path}")
        mmengine.dump(data, info_path)

    if val_infos:
        data = dict(data_list=val_infos, metainfo=metainfo)
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        print(f"Saving validation info to {info_val_path}")
        mmengine.dump(data, info_val_path)

def _fill_trainval_infos(root_path):
    """
    根据 trainval.yaml 划分训练集和验证集
    """
    # 加载 trainval.yaml
    config_file = osp.join(root_path, "trainval.yaml")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    train_scenes = config["train"]
    val_scenes = config["val"]

    train_infos = []
    val_infos = []
    num_frames  = 0

    # 遍历每个场景
    for scene in os.listdir(root_path):
        if not os.path.isdir(osp.join(root_path, scene)):
            continue
        if scene not in train_scenes and scene not in val_scenes:
            continue  # 跳过非训练/验证集的文件夹

        scene_dir = osp.join(root_path, scene)
        point_cloud_dir = osp.join(scene_dir, "lidar_point_cloud_0")
        label_dir = osp.join(scene_dir, "labels")
        camera_config_dir = osp.join(scene_dir, "camera_config")

        if not os.path.exists(point_cloud_dir) or not os.path.exists(label_dir) or not os.path.exists(camera_config_dir):
            print(f"Scene {scene} is missing required directories.")
            continue

        # 获取所有点云文件名
        points_files = os.listdir(point_cloud_dir)

        for file in points_files:
            file_name = os.path.splitext(file)[0]
            lidar_path = osp.join(point_cloud_dir, file_name + ".pcd")
            print(f"processing {lidar_path}")
            label_path = osp.join(label_dir, file_name + ".txt")
            calib_file_path = osp.join(camera_config_dir, file_name + ".json")

            mmengine.check_file_exist(lidar_path)
            mmengine.check_file_exist(label_path)
            mmengine.check_file_exist(calib_file_path)

            num_frames += 1
            # 构建 info 字典
            info = {
                'sample_idx': int(num_frames),
                'token': file_name,
                'lidar_points': {
                    'lidar_path': osp.relpath(lidar_path, root_path),
                    'num_pts_feats': 3,
                    'lidar2ego': np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ]),
                    'ego2global': None
                },
                'images': {},
                'instances': [],
                'cam_instances': {}
            }

            # 处理图像信息
            cameras = ['0', '1', '2', '3']
            for cam_name in cameras:
                cam_key = f"CAM_{cam_name}"
                img_path = osp.join(scene_dir, f"camera_image_{cam_name}", file_name + ".jpg")
                mmengine.check_file_exist(img_path)

                # 读取相机参数
                with open(calib_file_path, 'r') as f:
                    calib_data = json.load(f)

                P_values = calib_data[int(cam_name)]['camera_internal']
                RT_values = calib_data[int(cam_name)]['camera_external']

                cam_info = {
                    'img_path': osp.relpath(img_path, root_path),
                    'height': 1536,
                    'width': 1920,
                    'cam2img': np.array([
                        [P_values['fx'], 0.0, P_values['cx']],
                        [0.0, P_values['fy'], P_values['cy']],
                        [0.0, 0.0, 1.0]
                    ]),
                    'lidar2cam': np.array([
                        [RT_values[0], RT_values[1], RT_values[2], RT_values[3]],
                        [RT_values[4], RT_values[5], RT_values[6], RT_values[7]],
                        [RT_values[8], RT_values[9], RT_values[10], RT_values[11]],
                        [RT_values[12], RT_values[13], RT_values[14], RT_values[15]]
                    ])
                }
                info['images'][cam_key] = cam_info

            # 读取标签信息
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    ann = line.strip().split()
                    if len(ann) < 8:
                        continue

                    bbox_3d = [float(x) for x in ann[:7]]
                    label = categories.get(ann[7], -1)

                    if label == -1:
                        continue  # 跳过无效类别

                    instance = {
                        'bbox_3d': bbox_3d,
                        'bbox_label_3d': label,
                        'bbox_3d_isvalid': None,
                        'num_lidar_pts': None
                    }
                    info['instances'].append(instance)

            # 将样本添加到对应的集合中
            if scene in train_scenes:
                train_infos.append(info)
            elif scene in val_scenes:
                val_infos.append(info)
    print(num_frames)

    return train_infos, val_infos

if __name__ == '__main__':
    # 示例调用
    root_path = 'data/xbzl_data'
    create_custom_dataset_infos(root_path, 'custom')
    print("Done.")