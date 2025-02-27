from pathlib import Path

import cv2
import numpy as np
import open3d
from PIL import Image
from matplotlib import cm, pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector

from rock_texture_analyzer.base import BaseProcessor, mark_as_method, ManuallyProcessRequiredException, \
    mark_as_single_thread
from rock_texture_analyzer.get_bottom import _should_flip_based_on_z, _are_points_empty, _create_boundary_masks
from rock_texture_analyzer.get_top import _calculate_extended_bounds, _find_valid_clusters, _filter_side_points, \
    _calculate_final_boundaries
from rock_texture_analyzer.least_squares_adjustment_direction import least_squares_adjustment_direction
from rock_texture_analyzer.surface import surface_interpolate_2d
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

        boundary_mask, external_mask = _create_boundary_masks(points)
        boundary_points = points[boundary_mask]
        external_points = points[external_mask]

        if _are_points_empty(boundary_points, external_points):
            return

        if _should_flip_based_on_z(boundary_points, external_points):
            cloud.points = open3d.utility.Vector3dVector(-points)

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

        best_rotation = least_squares_adjustment_direction(points)

        # 8. 应用最佳旋转到整个点云
        rotated_points = points.dot(best_rotation.T)
        cloud.points = open3d.utility.Vector3dVector(rotated_points)
        write_point_cloud(output_path, cloud)

    @mark_as_method
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

        # 处理Z轴选择边界点
        point_z = points[:, 2]
        bottom_center = np.min(point_z)
        top_center = np.max(point_z)
        range_z = top_center - bottom_center
        z_selector = (point_z > (bottom_center + range_z * 0.1)) & (point_z < (top_center - range_z * 0.4))
        boundary_points = points[z_selector]
        point_x, point_y = boundary_points[:, 0], boundary_points[:, 1]

        # 通过阈值尝试获取有效聚类中心
        thresholds = [0.1, 0.05, 0.03, 0.02, 0.01]
        left_center, right_center, front_center, back_center = _find_valid_clusters(point_x, point_y, thresholds)

        self.print_safe(f'{left_center=} {right_center=}')
        self.print_safe(f'{front_center=} {back_center=}')
        assert back_center > front_center and right_center > left_center

        # 计算扩展边界范围
        (extend_x, extend_y,
         definite_left, definite_right,
         definite_front, definite_back) = _calculate_extended_bounds(
            left_center, right_center,
            front_center, back_center
        )

        # 筛选各边界点
        left_points = _filter_side_points(boundary_points, 0, left_center, extend_x, definite_front, definite_back)
        right_points = _filter_side_points(boundary_points, 0, right_center, extend_x, definite_front, definite_back)
        front_points = _filter_side_points(boundary_points, 1, front_center, extend_y, definite_left, definite_right)
        back_points = _filter_side_points(boundary_points, 1, back_center, extend_y, definite_left, definite_right)

        # 计算最终边界
        left, right, front, back = _calculate_final_boundaries(left_points, right_points, front_points, back_points)

        self.print_safe(f'{left=} {right=} {front=} {back=}')

        # 应用最终筛选条件
        point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]
        top_selector = (
                (point_x > left) & (point_x < right)
                & (point_y > front) & (point_y < back)
                & (point_z > bottom_center + range_z * 0.5)
        )
        cloud.points = open3d.utility.Vector3dVector(points[top_selector])

        # 处理颜色数据
        colors = np.asarray(cloud.colors)
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
        interpolated_matrix = surface_interpolate_2d(cloud, 0.1, 'cubic')

        layer_names = ['z', 'r', 'g', 'b']
        num_layers = interpolated_matrix.shape[-1]
        for i in range(num_layers):
            layer = interpolated_matrix[:, :, i]
            total = layer.size
            nan_count = np.isnan(layer).sum()
            nan_percentage = (nan_count / total) * 100
            layer_name = layer_names[i] if i < len(layer_names) else f'layer_{i}'
            self.print_safe(f"Layer '{layer_name}': {total=} {nan_count=} {nan_percentage=:.2f}%")

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


if __name__ == '__main__':
    s25022602_劈裂面形貌扫描_花岗岩_低曝光度.main()
