from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import open3d as o3d


class PointCloudProcessor:
    def __init__(self, base_dir: Path, project_name: str, grid_size: float = 0.1, threshold: int = 10):
        self.base_dir = base_dir
        self.project_name = project_name
        self.grid_size = grid_size
        self.threshold = threshold

    @cached_property
    def ply_files(self):
        """获取项目目录下的所有 PLY 文件"""
        return list(self.base_dir.glob(f'{self.project_name}/*.ply'))

    @cached_property
    def ply_file(self):
        """确保只有一个 PLY 文件存在并返回该文件"""
        if not self.ply_files:
            raise FileNotFoundError(f"No PLY files found in {self.base_dir / self.project_name}")
        return more_itertools.only(self.ply_files)

    @cached_property
    def point_cloud(self):
        """读取 PLY 点云文件"""
        return o3d.io.read_point_cloud(self.ply_file.as_posix())

    @cached_property
    def points(self):
        """提取点坐标"""
        return np.asarray(self.point_cloud.points)

    @cached_property
    def centroid(self):
        """计算点云的质心"""
        return np.mean(self.points, axis=0)

    @cached_property
    def centered_points(self):
        """中心化点云"""
        return self.points - self.centroid

    @cached_property
    def cov_matrix(self):
        """计算协方差矩阵"""
        return np.cov(self.centered_points, rowvar=False)

    @cached_property
    def plane_normal(self):
        """计算主体平面的法向量"""
        _, _, vh = np.linalg.svd(self.cov_matrix)
        normal = vh[-1]
        return normal / np.linalg.norm(normal)

    @cached_property
    def rotation_matrix(self):
        """计算将平面法向量对齐到目标法向量的旋转矩阵"""
        target_normal = np.array([0, 0, 1])
        v = np.cross(self.plane_normal, target_normal)
        s = np.linalg.norm(v)
        c = np.dot(self.plane_normal, target_normal)

        if s < 1e-6:
            # 法向量已经对齐，无需旋转
            return np.eye(3)
        else:
            # 计算旋转矩阵
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            return np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))

    @cached_property
    def rotated_points(self):
        """应用旋转矩阵后的点云"""
        return self.centered_points.dot(self.rotation_matrix.T)

    @cached_property
    def projected_points(self):
        """将点云投影到平面 (z=0)"""
        return self.rotated_points[:, :2]

    @cached_property
    def histogram_data(self):
        """计算2D直方图数据并过滤"""
        x_min, y_min = self.projected_points.min(axis=0)
        x_max, y_max = self.projected_points.max(axis=0)

        x_bins = np.arange(x_min, x_max + self.grid_size, self.grid_size)
        y_bins = np.arange(y_min, y_max + self.grid_size, self.grid_size)

        hist, x_edges, y_edges = np.histogram2d(
            self.projected_points[:, 0],
            self.projected_points[:, 1],
            bins=[x_bins, y_bins]
        )

        non_zero_cells = np.count_nonzero(hist)
        print(f"非零网格点数量: {non_zero_cells}")

        # 仅保留数量超过阈值的网格
        hist_filtered = np.where(hist > self.threshold, 1, np.nan)
        return hist_filtered, x_edges, y_edges

    def plot_density(self):
        """绘制密度图"""
        hist, x_edges, y_edges = self.histogram_data

        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x_edges, y_edges, hist.T)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')  # 保持比例
        plt.tight_layout()
        plt.show()

    @classmethod
    def main(cls):
        """主函数入口"""
        base_dir = Path(r'F:\data\laser-scanner')
        project_name = 'Group_4'

        processor = cls(base_dir, project_name)

        # 打印点云信息
        print(processor.point_cloud)
        print("点云中的点数:", len(processor.point_cloud.points))
        print("点云中的颜色数:", len(processor.point_cloud.colors))
        print(f"主体平面法向量: {processor.plane_normal}")

        if not np.allclose(processor.rotation_matrix, np.eye(3)):
            print(f"旋转矩阵:\n{processor.rotation_matrix}")

        # 绘制密度图
        processor.plot_density()


if __name__ == '__main__':
    PointCloudProcessor.main()
