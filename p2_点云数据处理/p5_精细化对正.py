import copy

import numpy as np
import open3d
from sci_cache import sci_method_cache
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from scripts.config import base_dir, project_name
from scripts.p4_调整地面在下方 import PointCloudProcessorP4


class PointCloudProcessorP5(PointCloudProcessorP4):

    @property
    @sci_method_cache
    def p5_优化精细对正(self):
        """
        细化对正，通过分别对X和Y轴进行K-Means聚类，扩展边界范围，并使用SciPy的优化方法旋转优化使四个侧边界与坐标轴对齐。
        """
        cloud = copy.deepcopy(self.p4_地面在下)
        points = np.asarray(cloud.points)

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
            [np.sin(gamma_rad),  np.cos(gamma_rad), 0],
            [0, 0, 1]
        ])
        best_rotation = R_z @ R_y @ R_x

        # 8. 应用最佳旋转到整个点云
        rotated_points = points.dot(best_rotation.T)
        cloud.points = open3d.utility.Vector3dVector(rotated_points)
        return cloud

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制点云(obj.p5_优化精细对正)


if __name__ == '__main__':
    PointCloudProcessorP5.main()
