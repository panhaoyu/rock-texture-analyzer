from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import open3d as o3d


class PointCloudProcessor:
    def __init__(self, base_dir: Path, project_name: str):
        self.base_dir = base_dir
        self.project_name = project_name
        self.point_cloud = self.load_point_cloud()

    def load_point_cloud(self) -> o3d.geometry.PointCloud:
        ply_files = list(self.base_dir.glob(f'{self.project_name}/*.ply'))
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {self.base_dir / self.project_name}")
        if len(ply_files) > 1:
            raise FileNotFoundError(
                f"Multiple PLY files found in {self.base_dir / self.project_name}, expected only one."
            )
        ply_file = more_itertools.only(ply_files)
        return o3d.io.read_point_cloud(str(ply_file))

    def adjust_main_plane(self):
        print(self.point_cloud)
        print("点云中的点数:", len(self.point_cloud.points))
        print("点云中的颜色数:", len(self.point_cloud.colors))

        points = np.asarray(self.point_cloud.points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        self.point_cloud.points = o3d.utility.Vector3dVector(centered_points)
        print("点云已中心化。")

        cov_matrix = np.cov(centered_points, rowvar=False)
        _, _, vh = np.linalg.svd(cov_matrix)
        plane_normal = vh[-1]
        plane_normal /= np.linalg.norm(plane_normal)
        print(f"主体平面法向量: {plane_normal}")

        target_normal = np.array([0, 0, 1])
        v = np.cross(plane_normal, target_normal)
        s = np.linalg.norm(v)
        c = np.dot(plane_normal, target_normal)

        if s < 1e-6:
            R = np.eye(3)
            print("法向量已与目标法向量对齐，无需旋转。")
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))
            print(f"旋转矩阵:\n{R}")

        rotated_points = centered_points.dot(R.T)
        self.point_cloud.points = o3d.utility.Vector3dVector(rotated_points)
        print("点云已旋转。")

    def plot_point_cloud(self):
        print("显示3D点云图...")
        o3d.visualization.draw_geometries([self.point_cloud], window_name="Point Cloud")
        print("3D点云图已显示。")

    def plot_density(self, grid_size: float, threshold: int):
        points = np.asarray(self.point_cloud.points)
        projected_points = points[:, :2]
        print("点云已投影到平面 (z=0)。")

        x_min, y_min = projected_points.min(axis=0)
        x_max, y_max = projected_points.max(axis=0)

        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        hist, x_edges, y_edges = np.histogram2d(
            projected_points[:, 0],
            projected_points[:, 1],
            bins=[x_bins, y_bins]
        )

        non_zero_cells = np.count_nonzero(hist)
        print(f"非零网格点数量: {non_zero_cells}")

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
        print("密度图已绘制。")

    def align_density_square(self, grid_size: float, threshold: int):
        points = np.asarray(self.point_cloud.points)
        projected_points = points[:, :2]

        x_min, y_min = projected_points.min(axis=0)
        x_max, y_max = projected_points.max(axis=0)

        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        hist, x_edges, y_edges = np.histogram2d(
            projected_points[:, 0],
            projected_points[:, 1],
            bins=[x_bins, y_bins]
        )

        hist_filtered = np.where(hist > threshold, 255, 0).astype(np.uint8)

        # Convert to binary image
        density_image = hist_filtered
        density_image = density_image[::-1]  # Flip vertically for correct orientation

        contours, _ = cv2.findContours(density_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("未检测到高密度区域。")
            return

        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]

        if angle < -45:
            angle = 90 + angle
        else:
            angle = angle

        print(f"检测到的旋转角度: {angle} degrees")

        # 创建3D旋转矩阵绕z轴
        theta = np.radians(-angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_z = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        # 应用旋转
        rotated_points = points.dot(R_z.T)
        self.point_cloud.points = o3d.utility.Vector3dVector(rotated_points)
        print("点云已根据密度图旋转以对齐方形边界。")

    @classmethod
    def main(cls):
        base_dir = Path(r'F:\data\laser-scanner')
        project_name = 'Group_4'
        grid_size = 0.1
        threshold = 10

        processor = cls(base_dir, project_name)
        print("点云加载完成。")

        processor.adjust_main_plane()
        processor.align_density_square(grid_size, threshold)
        processor.plot_density(grid_size, threshold)


if __name__ == '__main__':
    PointCloudProcessor.main()
