import numpy as np
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile
import os
# from pyntcloud import PyntCloud


@TRANSFORMS.register_module()
class CustomLoadPointsFromFile(LoadPointsFromFile):
    def __init__(self, load_dim=None, use_dim=None, **kwargs):
        super().__init__(**kwargs)
        if use_dim is None:
            self.use_dim = list(range(3))  # 默认使用前 3 个维度
        else:
            self.use_dim = use_dim
        self.load_dim = load_dim
    def _load_points(self, pts_filename: str) -> np.ndarray:
        """读取 pcd 文件， 得到 np.ndarray(N, 4)
        """
        with open(pts_filename, 'rb') as f:
            data = f.read()
            data_binary = data[data.find(b"DATA binary") + 12:]
            points = np.frombuffer(data_binary, dtype=np.float32).reshape(-1, self.load_dim)
            points = points.astype(np.float32)
        return points

    def transform(self, results: dict) -> dict:
        pts_file_path = results['lidar_path']
        if not os.path.exists(pts_file_path):
            raise FileNotFoundError(f"Lidar file not found: {pts_file_path}")

        points = self._load_points(pts_file_path)

        points = points[:, self.use_dim]

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])
        results['points'] = points

        return results