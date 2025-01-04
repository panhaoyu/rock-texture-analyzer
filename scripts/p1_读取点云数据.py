from pathlib import Path

import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import open3d
from open3d.cpu.pybind.geometry import PointCloud
from sci_cache import MethodDiskCache, sci_method_cache

from scripts.config import base_dir, project_name


class PointCloudProcessor(MethodDiskCache):
    def __init__(self, base_dir: Path, project_name: str):
        self.base_dir = base_dir
        self.project_name = project_name

        ply_files = list(self.base_dir.glob(f'{self.project_name}/*.ply'))
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {self.base_dir / self.project_name}")
        if len(ply_files) > 1:
            raise FileNotFoundError(
                f"Multiple PLY files found in {self.base_dir / self.project_name}, expected only one."
            )
        self.ply_file: Path = more_itertools.only(ply_files)

    def get_cache_folder(self) -> Path:
        return self.base_dir / self.project_name / 'cache'

    @property
    @sci_method_cache
    def p1_读取点云原始数据(self) -> PointCloud:
        return open3d.io.read_point_cloud(self.ply_file.as_posix())

    def 绘制点云(self, cloud: PointCloud):
        open3d.visualization.draw_geometries([cloud])

    def 绘制平面投影(self, cloud: PointCloud, plane: str, grid_size: float, threshold: int):
        points = np.asarray(cloud.points)
        match plane:
            case 'xOy':
                projected_points = points[:, :2]
            case 'xOz':
                projected_points = points[:, [0, 2]]
            case 'yOz':
                projected_points = points[:, 1:3]
            case _:
                raise ValueError(f"Invalid plane specified: {plane}")

        x_min, y_min = projected_points.min(axis=0)
        x_max, y_max = projected_points.max(axis=0)

        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        hist, x_edges, y_edges = np.histogram2d(
            projected_points[:, 0],
            projected_points[:, 1],
            bins=[x_bins, y_bins]
        )

        hist_filtered = np.where(hist > threshold, 1, np.nan)

        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x_edges, y_edges, hist_filtered.T, cmap='viridis')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Density Map')
        plt.axis('equal')
        plt.colorbar(label='Density')
        plt.tight_layout()
        plt.show()



    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制点云(obj.p1_读取点云原始数据)
        # obj.绘制点云(obj.p6_仅保留顶面)
        # obj.绘制平面投影(obj.p7_仅保留明确的矩形区域, 'xOy', 0.1, 1)
        # obj.绘制表面()


if __name__ == '__main__':
    PointCloudProcessor.main()
