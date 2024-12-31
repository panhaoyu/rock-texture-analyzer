from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import open3d
from PIL import Image
from open3d.cpu.pybind.geometry import PointCloud
from sci_cache import MethodDiskCache, sci_method_cache
from scipy.interpolate import griddata

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

    @property
    @sci_method_cache
    def p8_表面二维重建(self) -> np.ndarray:
        """
        将上表面的点云通过散点插值转换为 x, y 平面内的 [z, r, g, b] 四层矩阵。

        Args:
            resolution: (x_res, y_res)，表示插值后矩阵的大小，即 x 和 y 方向的分辨率。

        Returns:
            np.ndarray: 生成的矩阵，包含 [z, r, g, b] 四个层。
        """
        cloud = self.p6_仅保留顶面

        resolution: float = 0.1

        points = np.asarray(cloud.points)

        # 提取 x, y, z, r, g, b 数据
        x, y, z = points.T


        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        x_count = int((x_max - x_min) / resolution)
        y_count = int((y_max - y_min) / resolution)

        # 创建网格
        x_edge = np.linspace(x_min, x_max, x_count)
        y_edge = np.linspace(y_min, y_max, y_count)
        x_grid, y_grid = np.meshgrid(x_edge, y_edge)

        # 插值 z, r, g, b 数据
        arrays = []
        z_interp = griddata((x, y), z, (x_grid, y_grid), method='cubic')
        arrays.extend([z_interp])
        colors = np.asarray(cloud.colors)
        if colors.size:
            r, g, b = colors.T
            r_interp = griddata((x, y), r, (x_grid, y_grid), method='cubic')
            g_interp = griddata((x, y), g, (x_grid, y_grid), method='cubic')
            b_interp = griddata((x, y), b, (x_grid, y_grid), method='cubic')
            arrays.extend([r_interp, g_interp, b_interp])

        # 生成 [z, r, g, b] 四层矩阵
        interpolated_matrix = np.stack(arrays, axis=-1)

        return interpolated_matrix

    def 绘制表面(self):
        interpolated_matrix = self.p8_表面二维重建

        # 绘制高程图
        elevation = interpolated_matrix[:, :, 0]
        norm = plt.Normalize(vmin=np.nanmin(elevation), vmax=np.nanmax(elevation))
        scalar_map = cm.ScalarMappable(cmap='jet', norm=norm)
        elevation = (scalar_map.to_rgba(elevation)[:, :, :3] * 255)
        elevation = np.uint8(elevation)
        Image.fromarray(elevation).save(self.ply_file.with_name('elevation.png'))

        if interpolated_matrix.shape[2] > 1:
            color = interpolated_matrix[:, :, 1:]
            for i in range(3):
                v = color[:, :, i]
                v_min = np.nanmin(v)
                v_max = np.nanmax(v)
                v = (v - v_min) / (v_max - v_min) * 255
                color[:, :, i] = v

            color = np.uint8(np.round(color))
            Image.fromarray(color).save(self.ply_file.with_name('surface.png'))



    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制点云(obj.p1_读取点云原始数据)
        # obj.绘制点云(obj.p6_仅保留顶面)
        # obj.绘制平面投影(obj.p7_仅保留明确的矩形区域, 'xOy', 0.1, 1)
        # obj.绘制表面()


if __name__ == '__main__':
    PointCloudProcessor.main()
