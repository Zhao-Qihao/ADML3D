import numpy as np
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile
import mmengine


@TRANSFORMS.register_module()
class CustomLoadPointsFromPCDFile(LoadPointsFromFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """读取 pcd 文件，得到 np.ndarray(N,)的一维数组"""
        try:
            with open(pts_filename, 'rb') as f:
                data = f.read()
                
            # 查找二进制数据起始位置
            data_start = data.find(b"DATA binary")
            if data_start == -1:
                raise ValueError(f"Invalid PCD file format: {pts_filename}")
            
            data_binary = data[data_start + 12:]
            # 注意：返回一维数组，让父类的transform方法处理reshape
            points = np.frombuffer(data_binary, dtype=np.float32)

            # 检查数据维度
            if len(points) % self.load_dim != 0:
                raise ValueError(f"Point cloud data dimension mismatch. "
                               f"Expected multiple of {self.load_dim}, got {len(points)}")
            
            return points
            
        except Exception as e:
            mmengine.check_file_exist(pts_filename)
            raise RuntimeError(f"Failed to load PCD file {pts_filename}: {str(e)}")