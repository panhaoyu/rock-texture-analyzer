from pathlib import Path

import cv2
import numpy as np
import open3d
from PIL import Image
from matplotlib import cm, pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector
from scipy.interpolate import griddata
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from p3_表面显微镜扫描数据处理.base import BaseProcessor, mark_as_method, ManuallyProcessRequiredException, \
    mark_as_single_thread
from rock_texture_analyzer.utils.get_two_peaks import get_two_main_value_filtered, ValueDetectionError
from rock_texture_analyzer.utils.point_cloud import write_point_cloud, read_point_cloud, draw_point_cloud


class s25022602_劈裂面形貌扫描_花岗岩_低曝光度(BaseProcessor):
    @mark_as_method
    def f1_原始数据(self, output_path: Path) -> None:
        raise ManuallyProcessRequiredException

    @mark_as_method
    def f2_读取点云原始数据(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f1_原始数据, output_path)
        output_path = output_path.with_suffix('.ply')
        cloud = read_point_cloud(input_path)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f3_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f2_读取点云原始数据, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    def f4_调整为主平面(self, output_path: Path) -> None:
        output_path = output_path.with_suffix('.ply')
        cloud = read_point_cloud(self.get_input_path(self.f2_读取点云原始数据, output_path))
        points = np.asarray(cloud.points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        cloud.points = Vector3dVector(centered_points)

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
        cloud.points = Vector3dVector(rotated_points)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f5_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f4_调整为主平面, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    def f6_xOy平面对正(self, output_path: Path) -> None:
        output_path = output_path.with_suffix('.ply')
        grid_size = 1
        threshold = 50
        cloud = read_point_cloud(self.get_input_path(self.f4_调整为主平面, output_path))
        points = np.asarray(cloud.points)
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
        cloud.points = open3d.utility.Vector3dVector(rotated_points)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f7_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f6_xOy平面对正, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    @mark_as_single_thread
    def f8_调整地面在下(self, output_path: Path) -> None:
        output_path = output_path.with_suffix('.ply')
        cloud = read_point_cloud(self.get_input_path(self.f6_xOy平面对正, output_path))
        points = np.asarray(cloud.points)
        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1].reshape(-1, 1)

        kmeans_x: KMeans = KMeans(n_clusters=2, random_state=0)
        kmeans_x.fit(x)
        centers_x = sorted(kmeans_x.cluster_centers_.flatten())
        xmin, xmax = centers_x[0], centers_x[1]

        kmeans_y: KMeans = KMeans(n_clusters=2, random_state=0)
        kmeans_y.fit(y)
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
            cloud.points = open3d.utility.Vector3dVector(flipped_points)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f9_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f8_调整地面在下, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    @mark_as_single_thread
    def f10_精细化对正(self, output_path: Path) -> None:
        """
        细化对正，通过分别对X和Y轴进行K-Means聚类，扩展边界范围，并使用SciPy的优化方法旋转优化使四个侧边界与坐标轴对齐。
        """
        output_path = output_path.with_suffix('.ply')
        cloud = read_point_cloud(self.get_input_path(self.f8_调整地面在下, output_path))
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
            [np.sin(gamma_rad), np.cos(gamma_rad), 0],
            [0, 0, 1]
        ])
        best_rotation = R_z @ R_y @ R_x

        # 8. 应用最佳旋转到整个点云
        rotated_points = points.dot(best_rotation.T)
        cloud.points = open3d.utility.Vector3dVector(rotated_points)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f11_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f10_精细化对正, output_path)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    def f12_仅保留顶面(self, output_path: Path) -> None:
        """
        细化对正，通过分别对X和Y轴进行K-Means聚类，扩展边界范围，并使用SciPy的优化方法旋转优化使四个侧边界与坐标轴对齐。
        """
        output_path = output_path.with_suffix('.ply')
        cloud = read_point_cloud(self.get_input_path(self.f10_精细化对正, output_path))
        points = np.asarray(cloud.points)

        point_z = points[:, 2]
        bottom_center = np.min(point_z)
        top_center = np.max(point_z)
        range_z = (top_center - bottom_center)
        z_selector = (point_z > (bottom_center + range_z * 0.1)) & (point_z < (top_center - range_z * 0.4))
        boundary_points = points[z_selector]
        point_x, point_y = boundary_points[:, 0], boundary_points[:, 1]

        # 尝试不同的阈值进行处理
        thresholds = [0.1, 0.05, 0.03, 0.02, 0.01]
        for threshold in thresholds:
            try:
                left_center, right_center = get_two_main_value_filtered(point_x, threshold)
                front_center, back_center = get_two_main_value_filtered(point_y, threshold)
                break
            except ValueDetectionError:
                continue
        else:
            raise ValueDetectionError(f"无法找到有效阈值，尝试了所有阈值: {thresholds}")

        print(f'{left_center=} {right_center=}')
        print(f'{front_center=} {back_center=}')

        assert back_center > front_center
        assert right_center > left_center

        # 2. 扩展边界范围，向内外分别扩展10%
        extend_x = 0.1 * (right_center - left_center)
        extend_y = 0.1 * (back_center - front_center)

        definite_front = front_center + extend_y
        definite_back = back_center - extend_y
        definite_left = left_center + extend_x
        definite_right = right_center - extend_x

        left_points = boundary_points[np.abs(point_x - left_center) < extend_x]
        right_points = boundary_points[np.abs(point_x - right_center) < extend_x]
        front_points = boundary_points[np.abs(point_y - front_center) < extend_y]
        back_points = boundary_points[np.abs(point_y - back_center) < extend_y]

        left_points = left_points[(left_points[:, 1] > definite_front) & (left_points[:, 1] < definite_back)]
        right_points = right_points[(right_points[:, 1] > definite_front) & (right_points[:, 1] < definite_back)]
        front_points = front_points[(front_points[:, 0] > definite_left) & (front_points[:, 0] < definite_right)]
        back_points = back_points[(back_points[:, 0] > definite_left) & (back_points[:, 0] < definite_right)]

        left_mean = np.mean(left_points[:, 0])
        left_std = np.std(left_points[:, 0])
        right_mean = np.mean(right_points[:, 0])
        right_std = np.std(right_points[:, 0])
        front_mean = np.mean(front_points[:, 1])
        front_std = np.std(front_points[:, 1])
        back_mean = np.mean(back_points[:, 1])
        back_std = np.std(back_points[:, 1])

        std_range = 5
        left = left_mean + std_range * left_std
        right = right_mean - std_range * right_std
        front = front_mean + std_range * front_std
        back = back_mean - std_range * back_std

        print(f'{left=} {right=} {front=} {back=}')

        point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]
        top_selector = (
                (point_x > left) & (point_x < right)
                & (point_y > front) & (point_y < back)
                & (point_z > bottom_center + range_z * 0.5)
        )

        cloud.points = open3d.utility.Vector3dVector(points[top_selector])

        colors = np.asarray(cloud.colors)  # 扫描的时候未必开启对颜色的扫描
        if colors.size:
            cloud.colors = open3d.utility.Vector3dVector(colors[top_selector])
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f13_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f12_仅保留顶面, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    def f14_表面二维重建(self, output_path: Path) -> None:
        """
        使用指定的插值方法将点云数据插值到二维网格上。

        Args:
            method (str): 插值方法，例如 'nearest', 'linear', 'cubic'。

        Returns:
            np.ndarray: 插值后的 [z, r, g, b] 四层矩阵。
        """
        output_path = output_path.with_suffix('.npy')
        cloud = read_point_cloud(self.get_input_path(self.f12_仅保留顶面, output_path))

        resolution: float = 0.1
        method = 'cubic'

        points = np.asarray(cloud.points)

        # 提取 x, y, z 数据
        x, y, z = points.T

        x_min = np.min(x) + 0.2
        x_max = np.max(x) - 0.2
        y_min = np.min(y) + 0.2
        y_max = np.max(y) - 0.2

        # 创建网格
        x_edge = np.arange(x_min, x_max, resolution)
        y_edge = np.arange(y_min, y_max, resolution)
        x_grid, y_grid = np.meshgrid(x_edge, y_edge)

        # 插值 z, r, g, b 数据
        arrays = []
        z_interp = griddata((x, y), z, (x_grid, y_grid), method=method)
        arrays.append(z_interp)
        colors = np.asarray(cloud.colors)
        if colors.size:
            r, g, b = colors.T
            r_interp = griddata((x, y), r, (x_grid, y_grid), method=method)
            g_interp = griddata((x, y), g, (x_grid, y_grid), method=method)
            b_interp = griddata((x, y), b, (x_grid, y_grid), method=method)
            arrays.extend([r_interp, g_interp, b_interp])

        # 生成 [z, r, g, b] 四层矩阵
        interpolated_matrix = np.stack(arrays, axis=-1)

        layer_names = ['z', 'r', 'g', 'b']
        num_layers = interpolated_matrix.shape[-1]
        for i in range(num_layers):
            layer = interpolated_matrix[:, :, i]
            total = layer.size
            nan_count = np.isnan(layer).sum()
            nan_percentage = (nan_count / total) * 100
            layer_name = layer_names[i] if i < len(layer_names) else f'layer_{i}'
            print(f"Layer '{layer_name}':")
            print(f"  总元素数量 (Total elements) = {total}")
            print(f"  NaN 数量 (NaN count) = {nan_count}")
            print(f"  NaN 占比 (NaN percentage) = {nan_percentage:.2f}%\n")

        np.save(output_path, interpolated_matrix)

    @mark_as_method
    def f15_绘制高程(self, output_path: Path) -> None:
        """
        绘制表面高程图和颜色图，保存后根据是否存在颜色图像进行显示。
        如果存在颜色图像，则将高程图和颜色图左右合并并显示。
        如果不存在颜色图像，则仅显示高程图像。

        Args:
            interpolated_matrix (np.ndarray): 预先计算好的插值结果矩阵。
            name (str): 名称的前缀。
        """
        interpolated_matrix = self.get_input_array(self.f14_表面二维重建, output_path)

        # 绘制高程图
        elevation = interpolated_matrix[:, :, 0]
        norm = plt.Normalize(
            vmin=float(np.nanquantile(elevation, 0.01)),
            vmax=float(np.nanquantile(elevation, 0.99)),
        )
        scalar_map = cm.ScalarMappable(cmap='jet', norm=norm)
        elevation_rgba = scalar_map.to_rgba(elevation)
        elevation_rgb = (elevation_rgba[:, :, :3] * 255).astype(np.uint8)
        elevation_image = Image.fromarray(elevation_rgb)
        elevation_image.save(output_path)

    @mark_as_method
    def f16_绘制图像(self, output_path: Path) -> None:
        interpolated_matrix = self.get_input_array(self.f14_表面二维重建, output_path)
        if interpolated_matrix.shape[2] > 1:
            # 处理颜色数据
            color = interpolated_matrix[:, :, 1:].copy()
            for i in range(3):
                v = color[:, :, i]
                v_min = np.nanquantile(v, 0.01)
                v_max = np.nanquantile(v, 0.99)
                if v_max - v_min == 0:
                    normalized_v = np.zeros_like(v)
                else:
                    normalized_v = (v - v_min) / (v_max - v_min) * 255
                color[:, :, i] = normalized_v

            color_uint8 = np.clip(np.round(color), 0, 255).astype(np.uint8)
            surface_image = Image.fromarray(color_uint8)
            surface_image.save(output_path)

    @mark_as_method
    def f17_合并两张图(self, output_path: Path) -> None:
        elevation_image = self.get_input_image(self.f15_绘制高程, output_path)
        surface_image = self.get_input_image(self.f16_绘制图像, output_path)

        if surface_image:
            # 确保两张图像的尺寸相同
            if elevation_image.size != surface_image.size:
                raise ValueError("高程图和颜色图的尺寸不一致，无法合并显示。")

            # 创建一个新的空白图像，宽度为两张图像的总和，高度相同
            combined_width = elevation_image.width + surface_image.width
            combined_height = elevation_image.height
            combined_image = Image.new('RGB', (combined_width, combined_height))

            # 将高程图粘贴到左侧
            combined_image.paste(elevation_image, (0, 0))

            # 将颜色图粘贴到右侧
            combined_image.paste(surface_image, (elevation_image.width, 0))

            # 显示合并后的图像
            result = combined_image
        else:
            # 仅显示高程图像
            result = elevation_image
        result.save(output_path)

    def 绘制三维表面_matlab(self, array: np.ndarray):
        """
        使用 MATLAB 绘制三维表面高程图。

        Args:
            array (np.ndarray): 输入的三维矩阵，形状为 (M, N, 1) 或 (M, N, 4)。
        """
        # 只取第一层作为高程数据
        elevation = array[:, :, 0]
        elevation_min, elevation_max = np.nanquantile(elevation, 0.01), np.nanquantile(elevation, 0.99)
        elevation[elevation < elevation_min] = elevation_min
        elevation[elevation > elevation_max] = elevation_max

        # 获取矩阵大小
        M, N = elevation.shape

        resolution = self.grid_resolution

        # 生成x和y坐标，确保长度为 N 和 M
        x_edge = np.linspace(0, (N - 1) * resolution, N)
        y_edge = np.linspace(0, (M - 1) * resolution, M)
        x_grid, y_grid = np.meshgrid(x_edge, y_edge)

        # 确认网格和Z的形状一致
        if x_grid.shape != elevation.shape or y_grid.shape != elevation.shape:
            raise ValueError(f"网格和高程数据的形状不匹配: X={x_grid.shape}, Y={y_grid.shape}, Z={elevation.shape}")

        eng = matlab.engine.start_matlab()

        X = matlab.double(x_grid.tolist())
        Y = matlab.double(y_grid.tolist())
        Z = matlab.double(elevation.tolist())

        # 调用 MATLAB 绘图函数
        print("在 MATLAB 中绘制三维表面...")
        eng.figure(nargout=0)
        eng.mesh(X, Y, Z, nargout=0)
        eng.grid(nargout=0)

        output_dir = self.ply_file.with_name('images')
        output_dir.mkdir(parents=True, exist_ok=True)
        matlab_plot_path = str(output_dir.joinpath('matlab_surface_plot.png'))
        eng.savefig(matlab_plot_path, nargout=0)
        print(f"MATLAB 图像已保存到 {matlab_plot_path}")

        # 显示 MATLAB 图形窗口
        eng.show(nargout=0)

    def 绘制表面_导出到AutoCAD(self, array: np.ndarray):
        """
        将插值后的高程数据导出为 AutoCAD 支持的 DXF 文件。

        Args:
            array (np.ndarray): 插值后的 [z, r, g, b] 矩阵。
        """
        # 提取并处理高程数据
        resolution_mm = 2.0
        skip_ratio = int(resolution_mm / self.grid_resolution)
        elevation = array[::skip_ratio, ::skip_ratio, 0]
        elevation_min, elevation_max = np.nanquantile(elevation, 0.01), np.nanquantile(elevation, 0.99)
        elevation = np.clip(elevation, elevation_min, elevation_max)

        # 高度放缩系数
        scale_z = 1.0
        avg_z = np.nanmean(elevation)
        elevation = (elevation - avg_z) * scale_z + avg_z

        M, N = elevation.shape
        x_edge = np.linspace(0, (N - 1) * resolution_mm, N)
        y_edge = np.linspace(0, (M - 1) * resolution_mm, M)

        # 创建 DXF 文档
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        # Collect vertices and create a vertex map
        vertices = []
        vertex_map = {}
        idx = 0
        for i in range(M):
            for j in range(N):
                z_val = elevation[i, j]
                if not np.isnan(z_val):
                    x = x_edge[j]
                    y = y_edge[i]
                    vertices.append((x, y, z_val))
                    vertex_map[(i, j)] = idx
                    idx += 1

        # Add vertices and faces using edit_data
        mesh = msp.add_mesh()
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices.extend(vertices)
            for i in range(M - 1):
                for j in range(N - 1):
                    if ((i, j) in vertex_map and (i, j + 1) in vertex_map and
                            (i + 1, j) in vertex_map and (i + 1, j + 1) in vertex_map):
                        v0 = vertex_map[(i, j)]
                        v1 = vertex_map[(i, j + 1)]
                        v2 = vertex_map[(i + 1, j)]
                        v3 = vertex_map[(i + 1, j + 1)]
                        mesh_data.faces.append([v0, v1, v2])
                        mesh_data.faces.append([v2, v1, v3])

        # 保存 DXF 文件
        output_dir = self.ply_file.with_name('autocad_exports')
        output_dir.mkdir(parents=True, exist_ok=True)
        dxf_path = output_dir.joinpath('elevation.dxf')
        doc.saveas(str(dxf_path))
        print(f"高程数据已成功导出到 {dxf_path}")

    def p8_一阶梯度幅值(self) -> np.ndarray:
        """
        计算一阶梯度的幅值，并将结果缓存起来。

        Returns:
            np.ndarray: 一阶梯度幅值的二维数组。
        """
        # 提取高度数据
        elevation = self.p7_表面二维重建[:, :, 0]

        # 计算一阶梯度
        grad_y, grad_x = np.gradient(elevation)

        # 对一阶梯度进行高斯平滑
        sigma = self.p8_gaussian_sigma
        grad_x_smoothed = gaussian_filter(grad_x, sigma=sigma)
        grad_y_smoothed = gaussian_filter(grad_y, sigma=sigma)

        # 计算一阶梯度的幅值
        first_gradient_magnitude = np.hypot(grad_x_smoothed, grad_y_smoothed)

        return first_gradient_magnitude

    def p8_二阶梯度幅值(self) -> np.ndarray:
        """
        计算二阶梯度的幅值，并将结果缓存起来。

        Returns:
            np.ndarray: 二阶梯度幅值的二维数组。
        """
        # 提取高度数据
        elevation = self.p7_表面二维重建[:, :, 0]

        # 计算一阶梯度
        grad_y, grad_x = np.gradient(elevation)
        if sigma := self.p8_gaussian_sigma:
            grad_x = gaussian_filter(grad_x, sigma=sigma)
            grad_y = gaussian_filter(grad_y, sigma=sigma)

        # 计算二阶梯度
        d2f_dxx = np.gradient(grad_x, axis=1)
        d2f_dyy = np.gradient(grad_y, axis=0)
        if sigma := self.p8_gaussian_sigma:
            d2f_dxx = gaussian_filter(d2f_dxx, sigma=sigma)
            d2f_dyy = gaussian_filter(d2f_dyy, sigma=sigma)

        # 计算二阶梯度的大小
        # second_gradient_magnitude = np.abs(d2f_dxx) + np.abs(d2f_dyy)
        second_gradient_magnitude = np.hypot(d2f_dxx, d2f_dyy)

    def 颜色与高程的相关性(self):
        """
        主方法，执行表面绘制和梯度计算与保存。
        """

        # 获取插值后的表面数据
        interpolated_matrix = self.p7_表面二维重建  # 使用三次插值

        # 绘制并保存表面图像
        self.绘制表面(interpolated_matrix, name='surface')

        # 获取一阶梯度幅值
        first_gradient_magnitude = self.p8_一阶梯度幅值
        # 将一阶梯度幅值转换为 [z] 层的三维数组
        first_gradient_matrix = first_gradient_magnitude[:, :, np.newaxis]
        # 绘制并保存一阶梯度图像
        self.绘制表面(first_gradient_matrix, name='first_gradient')

        # 获取二阶梯度幅值
        second_gradient_magnitude = self.p8_二阶梯度幅值
        # 将二阶梯度幅值转换为 [z] 层的三维数组
        second_gradient_matrix = second_gradient_magnitude[:, :, np.newaxis]
        # 绘制并保存二阶梯度图像
        self.绘制表面(second_gradient_matrix, name='second_gradient')


if __name__ == '__main__':
    s25022602_劈裂面形貌扫描_花岗岩_低曝光度.main()
