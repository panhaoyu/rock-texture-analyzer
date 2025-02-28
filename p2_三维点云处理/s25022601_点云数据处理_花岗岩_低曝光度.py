from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import cm, pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector

from rock_texture_analyzer.base import BaseProcessor, mark_as_method, ManuallyProcessRequiredException, \
    mark_as_single_thread, mark_as_ply, mark_as_npy
from rock_texture_analyzer.boundary_processing import compute_extended_bounds, filter_points_by_axis, \
    compute_statistical_boundaries, create_boundary_masks
from rock_texture_analyzer.clustering import find_two_peaks, ValueDetectionError, \
    find_single_peak
from rock_texture_analyzer.interpolation import surface_interpolate_2d
from rock_texture_analyzer.optimization import least_squares_adjustment_direction
from rock_texture_analyzer.other_utils import should_flip_based_on_z, compute_rotation_matrix


class s25022602_劈裂面形貌扫描_花岗岩_低曝光度(BaseProcessor):
    is_debug = False

    @mark_as_method
    def f1_原始数据(self, output_path: Path):
        raise ManuallyProcessRequiredException

    @mark_as_method
    @mark_as_ply
    def f2_读取点云原始数据(self, output_path: Path):
        cloud = self.get_input_ply(self.f1_原始数据, output_path)
        return cloud

    @mark_as_method
    @mark_as_single_thread
    def f3_绘制点云(self, output_path: Path):
        return self.get_input_ply(self.f2_读取点云原始数据, output_path)

    @mark_as_method
    @mark_as_ply
    def f4_调整为主平面(self, output_path: Path):
        cloud = self.get_input_ply(self.f2_读取点云原始数据, output_path)
        points = np.asarray(cloud.points)
        centered_points = points - np.mean(points, axis=0)
        plane_normal = np.linalg.svd(np.cov(centered_points.T))[2][-1]
        plane_normal /= np.linalg.norm(plane_normal)
        rotation_matrix = compute_rotation_matrix(plane_normal, [0, 0, 1])
        cloud.points = Vector3dVector(centered_points @ rotation_matrix.T)
        return cloud

    @mark_as_method
    @mark_as_single_thread
    def f5_绘制点云(self, output_path: Path):
        return self.get_input_ply(self.f4_调整为主平面, output_path)

    @mark_as_method
    @mark_as_ply
    def f6_xOy平面对正(self, output_path: Path):
        cloud = self.get_input_ply(self.f4_调整为主平面, output_path)
        points = np.asarray(cloud.points)
        x, y = points[:, :2].T
        x_bins = np.arange(x.min(), x.max() + 1)
        y_bins = np.arange(y.min(), y.max() + 1)
        hist = np.histogram2d(x, y, bins=[x_bins, y_bins])[0]
        density_image = np.where(hist > 50, 255, 0).astype(np.uint8)[::-1]

        if (contours := cv2.findContours(density_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]):
            angle = (rect := cv2.minAreaRect(max(contours, key=cv2.contourArea)))[-1]
            angle = angle if angle < -45 else angle + 90
            theta = np.radians(-angle)
            R_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            cloud.points = Vector3dVector(points @ R_z.T)

        return cloud

    @mark_as_method
    @mark_as_single_thread
    def f7_绘制点云(self, output_path: Path):
        return self.get_input_ply(self.f6_xOy平面对正, output_path)

    @mark_as_method
    @mark_as_ply
    @mark_as_single_thread
    def f8_调整地面在下(self, output_path: Path):
        cloud = self.get_input_ply(self.f6_xOy平面对正, output_path)
        points = np.asarray(cloud.points)
        if should_flip_based_on_z(*create_boundary_masks(points, extension_ratio=0.1)):
            cloud.points = Vector3dVector(-points)
        return cloud

    @mark_as_method
    @mark_as_single_thread
    def f9_绘制点云(self, output_path: Path):
        return self.get_input_ply(self.f8_调整地面在下, output_path)

    @mark_as_method
    @mark_as_ply
    @mark_as_single_thread
    def f10_精细化对正(self, output_path: Path):
        cloud = self.get_input_ply(self.f8_调整地面在下, output_path)
        points = np.asarray(cloud.points)
        best_rotation = least_squares_adjustment_direction(points)
        rotated_points = points.dot(best_rotation.T)
        cloud.points = Vector3dVector(rotated_points)
        return cloud

    @mark_as_method
    @mark_as_single_thread
    def f11_绘制点云(self, output_path: Path):
        return self.get_input_ply(self.f10_精细化对正, output_path)

    @mark_as_method
    def f11_计算顶面与底面位置的KDE图(self, output_path: Path):
        figure: plt.Figure = plt.figure()
        cloud = self.get_input_ply(self.f10_精细化对正, output_path)
        z = np.asarray(cloud.points)[:, 2]
        ax = figure.subplots()
        sns.kdeplot(z, fill=True, ax=ax)
        return figure

    @mark_as_method
    @mark_as_ply
    @mark_as_single_thread
    def f12_仅保留顶面(self, output_path: Path):
        cloud = self.get_input_ply(self.f10_精细化对正, output_path)
        points = np.asarray(cloud.points)
        point_z = points[:, 2]
        thresholds = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
        try:
            bottom, top = find_two_peaks(point_z, thresholds)
        except ValueDetectionError:
            bottom = np.min(point_z)
            top = find_single_peak(point_z, thresholds)
        self.print_safe(f'{bottom=} {top=}')
        range_z = top - bottom
        z_selector = (point_z > (bottom + range_z * 0.1)) & (point_z < (top - range_z * 0.4))
        boundary_points = points[z_selector]
        point_x, point_y = boundary_points[:, 0], boundary_points[:, 1]
        left_center, right_center = find_two_peaks(point_x, prominence=thresholds)
        front_center, back_center = find_two_peaks(point_y, prominence=thresholds)
        self.print_safe(f'{left_center=} {right_center=}')
        self.print_safe(f'{front_center=} {back_center=}')
        assert back_center > front_center and right_center > left_center
        (extend_x, extend_y, definite_left, definite_right, definite_front, definite_back) = \
            (compute_extended_bounds(left_center, right_center, front_center, back_center))
        left_points = filter_points_by_axis(boundary_points, 0, left_center, extend_x, definite_front, definite_back)
        right_points = filter_points_by_axis(boundary_points, 0, right_center, extend_x, definite_front, definite_back)
        front_points = filter_points_by_axis(boundary_points, 1, front_center, extend_y, definite_left, definite_right)
        back_points = filter_points_by_axis(boundary_points, 1, back_center, extend_y, definite_left, definite_right)
        left, right, front, back = compute_statistical_boundaries(left_points, right_points, front_points, back_points)
        self.print_safe(f'{left=} {right=} {front=} {back=}')
        point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]
        top_selector = (
                (point_x > left) & (point_x < right)
                & (point_y > front) & (point_y < back)
                & (point_z > (bottom + range_z * 0.5))
        )
        cloud.points = Vector3dVector(points[top_selector])
        colors = np.asarray(cloud.colors)
        if colors.size:
            cloud.colors = Vector3dVector(colors[top_selector])
        return cloud

    @mark_as_method
    @mark_as_single_thread
    def f13_绘制点云(self, output_path: Path):
        return self.get_input_ply(self.f12_仅保留顶面, output_path)

    @mark_as_method
    @mark_as_npy
    def f14_表面二维重建(self, output_path: Path):
        cloud = self.get_input_ply(self.f12_仅保留顶面, output_path)
        interpolated_matrix = surface_interpolate_2d(cloud, 0.1, 'cubic')
        for i, name in enumerate(['z', 'r', 'g', 'b'][:interpolated_matrix.shape[2]]):
            layer = interpolated_matrix[..., i]
            total, nan = layer.size, np.isnan(layer).sum()
            self.print_safe(f"Layer '{name}': {total=} {nan=} {nan / total * 100:.2f}%")
        return interpolated_matrix

    @mark_as_method
    def f15_绘制高程(self, output_path: Path):
        elevation = self.get_input_array(self.f14_表面二维重建, output_path)[..., 0]
        norm = plt.Normalize(*np.nanquantile(elevation, [0.01, 0.99]))
        return Image.fromarray(
            (cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(elevation)[..., :3] * 255).astype(np.uint8))

    @mark_as_method
    def f16_绘制图像(self, output_path: Path) -> np.ndarray:
        matrix = self.get_input_array(self.f14_表面二维重建, output_path)
        matrix = matrix[:, :, 1:4]
        assert matrix.shape[2] >= 3, "输入矩阵需要至少3个通道"

        def _normalize_channel(channel: np.ndarray) -> np.ndarray:
            v_min = np.nanquantile(channel, 0.01)
            v_max = np.nanquantile(channel, 0.99)
            return np.nan_to_num(
                (channel - v_min) / max(v_max - v_min, 1e-9),
                copy=False
            )

        channels = [
            _normalize_channel(matrix[..., i]) * 255
            for i in range(3)
        ]
        return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)

    @mark_as_method
    def f17_合并两张图(self, output_path: Path):
        """将高程图与表面图合并为横向排列的图片"""
        elevation_img = self.get_input_image(self.f15_绘制高程, output_path)
        surface_img = self.get_input_image(self.f16_绘制图像, output_path)

        if not surface_img:
            elevation_img.save(output_path)
            return

        if (size := elevation_img.size) != surface_img.size:
            raise ValueError("高程图与表面图尺寸不一致")

        combined_img = Image.new('RGB', (size[0] * 2, size[1]))
        combined_img.paste(elevation_img, (0, 0))
        combined_img.paste(surface_img, (size[0], 0))
        return combined_img


if __name__ == '__main__':
    s25022602_劈裂面形貌扫描_花岗岩_低曝光度.main()
