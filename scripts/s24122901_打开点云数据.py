from pathlib import Path

import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import open3d as o3d


class PointCloudProcessor:
    """点云处理器类，用于加载、处理和可视化PLY点云文件。

    Attributes:
        base_dir (Path): 基础目录路径。
        project_name (str): 项目名称，用于定位PLY文件。
        grid_size (float): 网格大小，用于直方图计算。
        threshold (int): 直方图过滤阈值。
        point_cloud (o3d.geometry.PointCloud): 当前点云数据。
    """

    def __init__(self, base_dir: Path, project_name: str, grid_size: float = 0.1, threshold: int = 10):
        """初始化点云处理器。

        Args:
            base_dir (Path): 基础目录路径。
            project_name (str): 项目名称，用于定位PLY文件。
            grid_size (float, optional): 网格大小，用于直方图计算。默认为0.1。
            threshold (int, optional): 直方图过滤阈值。默认为10。
        """
        self.base_dir = base_dir
        self.project_name = project_name
        self.grid_size = grid_size
        self.threshold = threshold
        self.point_cloud = self.load_point_cloud()

    def load_point_cloud(self) -> o3d.geometry.PointCloud:
        """加载项目目录下的单个PLY点云文件。

        Returns:
            o3d.geometry.PointCloud: 加载的点云对象。

        Raises:
            FileNotFoundError: 如果没有找到PLY文件或找到多个PLY文件。
        """
        ply_files = list(self.base_dir.glob(f'{self.project_name}/*.ply'))
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {self.base_dir / self.project_name}")
        if len(ply_files) > 1:
            raise FileNotFoundError(
                f"Multiple PLY files found in {self.base_dir / self.project_name}, expected only one."
            )
        ply_file = more_itertools.only(ply_files)
        point_cloud = o3d.io.read_point_cloud(str(ply_file))
        return point_cloud

    def process_point_cloud(self):
        """处理点云数据，包括中心化和旋转对齐。

        1. 打印点云基本信息。
        2. 计算点云的质心并中心化点云。
        3. 计算协方差矩阵并通过奇异值分解得到主体平面的法向量。
        4. 计算将平面法向量对齐到z轴的旋转矩阵并应用旋转。
        """
        # 打印点云信息
        print(self.point_cloud)
        print("点云中的点数:", len(self.point_cloud.points))
        print("点云中的颜色数:", len(self.point_cloud.colors))

        # 提取点坐标
        points = np.asarray(self.point_cloud.points)

        # 计算质心并中心化
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        self.point_cloud.points = o3d.utility.Vector3dVector(centered_points)
        print("点云已中心化。")

        # 计算协方差矩阵并求法向量
        cov_matrix = np.cov(centered_points, rowvar=False)
        _, _, vh = np.linalg.svd(cov_matrix)
        plane_normal = vh[-1]
        plane_normal /= np.linalg.norm(plane_normal)
        print(f"主体平面法向量: {plane_normal}")

        # 计算旋转矩阵
        target_normal = np.array([0, 0, 1])
        v = np.cross(plane_normal, target_normal)
        s = np.linalg.norm(v)
        c = np.dot(plane_normal, target_normal)

        if s < 1e-6:
            # 法向量已经对齐，无需旋转
            R = np.eye(3)
            print("法向量已与目标法向量对齐，无需旋转。")
        else:
            # 计算旋转矩阵
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))
            print(f"旋转矩阵:\n{R}")

        # 应用旋转
        rotated_points = centered_points.dot(R.T)
        self.point_cloud.points = o3d.utility.Vector3dVector(rotated_points)
        print("点云已旋转。")

    def plot_point_cloud(self):
        """显示当前的3D点云图。

        使用Open3D的可视化工具显示点云。
        """
        print("显示3D点云图...")
        o3d.visualization.draw_geometries([self.point_cloud], window_name="Point Cloud")
        print("3D点云图已显示。")

    def plot_density(self):
        """计算投影并绘制2D密度图，仅显示数量超过阈值的网格。

        使用Matplotlib绘制过滤后的2D直方图密度图。
        """
        # 提取点坐标
        points = np.asarray(self.point_cloud.points)

        # 投影到平面 (z=0)
        projected_points = points[:, :2]
        print("点云已投影到平面 (z=0)。")

        # 计算2D直方图
        x_min, y_min = projected_points.min(axis=0)
        x_max, y_max = projected_points.max(axis=0)

        x_bins = np.arange(x_min, x_max + self.grid_size, self.grid_size)
        y_bins = np.arange(y_min, y_max + self.grid_size, self.grid_size)

        hist, x_edges, y_edges = np.histogram2d(
            projected_points[:, 0],
            projected_points[:, 1],
            bins=[x_bins, y_bins]
        )

        non_zero_cells = np.count_nonzero(hist)
        print(f"非零网格点数量: {non_zero_cells}")

        # 仅保留数量超过阈值的网格
        hist_filtered = np.where(hist > self.threshold, 1, np.nan)

        # 绘制密度图
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x_edges, y_edges, hist_filtered.T, cmap='viridis')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Density Map')
        plt.axis('equal')  # 保持比例
        plt.colorbar(label='Density')
        plt.tight_layout()
        plt.show()
        print("密度图已绘制。")

    @classmethod
    def main(cls):
        """主函数入口，按步骤执行点云处理和绘图。

        Steps:
            1. 初始化处理器并加载点云。
            2. 处理点云数据。
            3. 显示处理后的3D点云。
            4. 绘制2D密度图。
        """
        base_dir = Path(r'F:\data\laser-scanner')
        project_name = 'Group_4'
        grid_size = 0.1
        threshold = 10

        # 初始化处理器
        processor = cls(base_dir, project_name, grid_size, threshold)
        print("点云加载完成。")

        # 处理点云
        processor.process_point_cloud()

        # 显示处理后的3D点云
        processor.plot_point_cloud()

        # 绘制密度图
        processor.plot_density()


if __name__ == '__main__':
    PointCloudProcessor.main()
