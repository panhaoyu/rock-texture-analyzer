from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from sklearn.cluster import KMeans


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
        points = np.asarray(self.point_cloud.points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        self.point_cloud.points = o3d.utility.Vector3dVector(centered_points)

        cov_matrix = np.cov(centered_points, rowvar=False)
        _, _, vh = np.linalg.svd(cov_matrix)
        plane_normal = vh[-1]
        plane_normal /= np.linalg.norm(plane_normal)

        target_normal = np.array([0, 0, 1])
        v = np.cross(plane_normal, target_normal)
        s = np.linalg.norm(v)
        c = np.dot(plane_normal, target_normal)

        if s < 1e-6:
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))

        rotated_points = centered_points.dot(R.T)
        self.point_cloud.points = o3d.utility.Vector3dVector(rotated_points)

    def plot_point_cloud(self):
        o3d.visualization.draw_geometries([self.point_cloud], window_name="Point Cloud")

    def plot_density(self, plane: str, grid_size: float, threshold: int):
        points = np.asarray(self.point_cloud.points)
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
        density_image = hist_filtered
        density_image = density_image[::-1]

        contours, _ = cv2.findContours(density_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]

        if angle < -45:
            angle = 90 + angle
        else:
            angle = angle

        theta = np.radians(-angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_z = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        rotated_points = points.dot(R_z.T)
        self.point_cloud.points = o3d.utility.Vector3dVector(rotated_points)

    def evaluate_and_flip_z(self):
        points = np.asarray(self.point_cloud.points)
        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1].reshape(-1, 1)

        kmeans_x = KMeans(n_clusters=2, random_state=0).fit(x)
        centers_x = sorted(kmeans_x.cluster_centers_.flatten())
        xmin, xmax = centers_x[0], centers_x[1]

        kmeans_y = KMeans(n_clusters=2, random_state=0).fit(y)
        centers_y = sorted(kmeans_y.cluster_centers_.flatten())
        ymin, ymax = centers_y[0], centers_y[1]

        extend_x = 0.1 * (xmax - xmin)
        extend_y = 0.1 * (ymax - ymin)

        xmin_ext = xmin - extend_x
        xmax_ext = xmax + extend_x
        ymin_ext = ymin - extend_y
        ymax_ext = ymax + extend_y

        boundary_mask = (
                (points[:, 0] >= xmin_ext) & (points[:, 0] <= xmax_ext) &
                (points[:, 1] >= ymin_ext) & (points[:, 1] <= ymax_ext)
        )

        boundary_points = points[boundary_mask]
        external_mask = (
                ((points[:, 0] > xmax_ext) | (points[:, 1] > ymax_ext)) &
                (points[:, 0] >= xmin_ext)
        )
        external_points = points[external_mask]

        if len(boundary_points) == 0 or len(external_points) == 0:
            return

        median_z_inside = np.median(boundary_points[:, 2])
        median_z_outside = np.median(external_points[:, 2])

        if median_z_outside > median_z_inside:
            flipped_points = points.copy()
            flipped_points[:, 2] = -flipped_points[:, 2]
            self.point_cloud.points = o3d.utility.Vector3dVector(flipped_points)

    def fine_align(self):
        """
        细化对正，通过分别对X和Y轴进行K-Means聚类，扩展边界范围，并使用SciPy的优化方法旋转优化使四个侧边界与坐标轴对齐。
        """
        points = np.asarray(self.point_cloud.points)

        # 1. 分别对X轴和Y轴进行K-Means聚类
        # 对X轴聚类
        kmeans_x = KMeans(n_clusters=2, random_state=0)
        kmeans_x.fit(points[:, 0].reshape(-1, 1))
        centers_x = sorted(kmeans_x.cluster_centers_.flatten())
        xmin, xmax = centers_x[0], centers_x[1]

        # 对Y轴聚类
        kmeans_y = KMeans(n_clusters=2, random_state=0)
        kmeans_y.fit(points[:, 1].reshape(-1, 1))
        centers_y = sorted(kmeans_y.cluster_centers_.flatten())
        ymin, ymax = centers_y[0], centers_y[1]

        # 2. 扩展边界范围，向内外分别扩展10%
        range_x = xmax - xmin
        range_y = ymax - ymin
        extend_x = 0.1 * range_x
        extend_y = 0.1 * range_y

        # 左侧边界
        left_mask = (points[:, 0] <= (xmin + extend_x)) & (points[:, 0] >= (xmin - extend_x))
        left_boundary = points[left_mask]

        # 右侧边界
        right_mask = (points[:, 0] >= (xmax - extend_x)) & (points[:, 0] <= (xmax + extend_x))
        right_boundary = points[right_mask]

        # 前侧边界
        front_mask = (points[:, 1] <= (ymin + extend_y)) & (points[:, 1] >= (ymin - extend_y))
        front_boundary = points[front_mask]

        # 后侧边界
        back_mask = (points[:, 1] >= (ymax - extend_y)) & (points[:, 1] <= (ymax + extend_y))
        back_boundary = points[back_mask]

        # 3. 分别处理每个边界
        boundary_left = left_boundary.copy()
        boundary_right = right_boundary.copy()
        boundary_front = front_boundary.copy()
        boundary_back = back_boundary.copy()

        # 4. 在高度方向上舍弃10%的点（顶部和底部各5%）
        def filter_height(boundary):
            if len(boundary) == 0:
                return boundary
            z_sorted = np.sort(boundary[:, 2])
            lower_bound = z_sorted[int(0.05 * len(z_sorted))]
            upper_bound = z_sorted[int(0.95 * len(z_sorted))]
            return boundary[
                (boundary[:, 2] >= lower_bound) &
                (boundary[:, 2] <= upper_bound)
                ]

        boundary_left = filter_height(boundary_left)
        boundary_right = filter_height(boundary_right)
        boundary_front = filter_height(boundary_front)
        boundary_back = filter_height(boundary_back)

        # 确保每个边界都有足够的点
        if any(len(b) == 0 for b in [boundary_left, boundary_right, boundary_front, boundary_back]):
            print("某些边界在高度过滤后没有剩余的点。")
            return

        # 5. 定义优化目标函数
        def objective(angles_deg):
            alpha, beta, gamma = angles_deg  # 旋转角度（度）
            # 转换为弧度
            alpha_rad = np.radians(alpha)
            beta_rad = np.radians(beta)
            gamma_rad = np.radians(gamma)

            # 构建旋转矩阵（顺序：X -> Y -> Z）
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(alpha_rad), -np.sin(alpha_rad)],
                [0, np.sin(alpha_rad), np.cos(alpha_rad)]
            ])
            R_y = np.array([
                [np.cos(beta_rad), 0, np.sin(beta_rad)],
                [0, 1, 0],
                [-np.sin(beta_rad), 0, np.cos(beta_rad)]
            ])
            R_z = np.array([
                [np.cos(gamma_rad), -np.sin(gamma_rad), 0],
                [np.sin(gamma_rad), np.cos(gamma_rad), 0],
                [0, 0, 1]
            ])
            R = R_z @ R_y @ R_x

            # 应用旋转
            rotated_left = boundary_left.dot(R.T)
            rotated_right = boundary_right.dot(R.T)
            rotated_front = boundary_front.dot(R.T)
            rotated_back = boundary_back.dot(R.T)

            # 计算标准差
            std_left = np.std(rotated_left[:, 0])  # 左侧边界关注x值
            std_right = np.std(rotated_right[:, 0])  # 右侧边界关注x值
            std_front = np.std(rotated_front[:, 1])  # 前侧边界关注y值
            std_back = np.std(rotated_back[:, 1])  # 后侧边界关注y值

            # 总目标：最小化所有标准差的加权和
            total_std = std_left + std_right + std_front + std_back

            return total_std

        # 6. 使用SciPy的minimize进行优化
        initial_angles = [0.0, 0.0, 0.0]  # 初始猜测角度（度）
        bounds = [(-10, 10), (-10, 10), (-10, 10)]  # 旋转角度范围（度）

        result = minimize(
            objective,
            initial_angles,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )

        if result.success:
            best_angles = result.x
            best_std = result.fun
            print(f"最佳旋转角度 (α, β, γ): {best_angles} 度, 总标准差: {best_std:.6f}")
        else:
            print("优化未收敛，使用初始角度。")
            best_angles = initial_angles

        # 7. 构建最佳旋转矩阵
        alpha, beta, gamma = best_angles  # 旋转角度（度）
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(alpha_rad), -np.sin(alpha_rad)],
            [0, np.sin(alpha_rad), np.cos(alpha_rad)]
        ])
        R_y = np.array([
            [np.cos(beta_rad), 0, np.sin(beta_rad)],
            [0, 1, 0],
            [-np.sin(beta_rad), 0, np.cos(beta_rad)]
        ])
        R_z = np.array([
            [np.cos(gamma_rad), -np.sin(gamma_rad), 0],
            [np.sin(gamma_rad), np.cos(gamma_rad), 0],
            [0, 0, 1]
        ])
        best_rotation = R_z @ R_y @ R_x

        # 8. 应用最佳旋转到整个点云
        rotated_points = points.dot(best_rotation.T)
        self.point_cloud.points = o3d.utility.Vector3dVector(rotated_points)

    @classmethod
    def main(cls):
        base_dir = Path(r'F:\data\laser-scanner')
        project_name = 'Group_4'
        processor = cls(base_dir, project_name)
        processor.adjust_main_plane()
        processor.align_density_square(grid_size=1, threshold=50)
        processor.evaluate_and_flip_z()
        processor.plot_density('xOy', grid_size=0.1, threshold=10)
        processor.fine_align()
        processor.plot_density('xOy', grid_size=0.1, threshold=10)


if __name__ == '__main__':
    PointCloudProcessor.main()
