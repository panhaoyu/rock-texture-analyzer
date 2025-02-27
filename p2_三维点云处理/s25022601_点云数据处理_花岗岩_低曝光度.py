from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from matplotlib import cm, pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector

from rock_texture_analyzer.base import BaseProcessor, mark_as_method, ManuallyProcessRequiredException, \
    mark_as_single_thread, mark_as_ply, mark_as_npy
from rock_texture_analyzer.boundary_processing import compute_extended_bounds, filter_points_by_axis, \
    compute_statistical_boundaries, create_boundary_masks
from rock_texture_analyzer.clustering import find_valid_clusters
from rock_texture_analyzer.interpolation import surface_interpolate_2d
from rock_texture_analyzer.optimization import least_squares_adjustment_direction
from rock_texture_analyzer.other_utils import should_flip_based_on_z
from rock_texture_analyzer.utils.point_cloud import write_point_cloud, read_point_cloud, draw_point_cloud


def compute_rotation_matrix(plane_normal: np.ndarray, target_normal: np.ndarray) -> np.ndarray:
    """计算将平面法向量旋转到目标法向量的旋转矩阵"""
    v = np.cross(plane_normal, target_normal)
    s, c = np.linalg.norm(v), np.dot(plane_normal, target_normal)
    if s < 1e-6:
        return np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))


class s25022602_劈裂面形貌扫描_花岗岩_低曝光度(BaseProcessor):
    is_debug = True

    @mark_as_method
    def f1_原始数据(self, output_path: Path) -> None:
        raise ManuallyProcessRequiredException

    @mark_as_method
    @mark_as_ply
    def f2_读取点云原始数据(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f1_原始数据, output_path)
        write_point_cloud(output_path, read_point_cloud(input_path))

    @mark_as_method
    @mark_as_single_thread
    def f3_绘制点云(self, output_path: Path) -> None:
        draw_point_cloud(self.get_file_path(self.f2_读取点云原始数据, output_path.stem), output_path)

    @mark_as_method
    @mark_as_ply
    def f4_调整为主平面(self, output_path: Path) -> None:
        cloud = read_point_cloud(self.get_input_path(self.f2_读取点云原始数据, output_path))
        points = np.asarray(cloud.points)
        centered_points = points - np.mean(points, axis=0)
        plane_normal = np.linalg.svd(np.cov(centered_points.T))[2][-1]
        plane_normal /= np.linalg.norm(plane_normal)

        cloud.points = Vector3dVector(centered_points @ compute_rotation_matrix(plane_normal, [0, 0, 1]).T)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f5_绘制点云(self, output_path: Path) -> None:
        draw_point_cloud(self.get_file_path(self.f4_调整为主平面, output_path.stem), output_path)

    @mark_as_method
    @mark_as_ply
    def f6_xOy平面对正(self, output_path: Path) -> None:
        cloud = read_point_cloud(self.get_input_path(self.f4_调整为主平面, output_path))
        points = np.asarray(cloud.points)
        x, y = points[:, :2].T
        x_bins = np.arange(x.min(), x.max() + 1, 1)
        y_bins = np.arange(y.min(), y.max() + 1, 1)

        hist = np.histogram2d(x, y, bins=[x_bins, y_bins])[0]
        density_image = np.where(hist > 50, 255, 0).astype(np.uint8)[::-1]

        if (contours := cv2.findContours(density_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]):
            angle = (rect := cv2.minAreaRect(max(contours, key=cv2.contourArea)))[-1]
            angle = angle if angle < -45 else angle + 90
            theta = np.radians(-angle)
            R_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            cloud.points = Vector3dVector(points @ R_z.T)

        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f7_绘制点云(self, output_path: Path) -> None:
        draw_point_cloud(self.get_file_path(self.f6_xOy平面对正, output_path.stem), output_path)

    @mark_as_method
    @mark_as_single_thread
    @mark_as_ply
    def f8_调整地面在下(self, output_path: Path) -> None:
        cloud = read_point_cloud(self.get_input_path(self.f6_xOy平面对正, output_path))
        points = np.asarray(cloud.points)
        if should_flip_based_on_z(*create_boundary_masks(points, extension_ratio=0.1)):
            cloud.points = Vector3dVector(-points)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f9_绘制点云(self, output_path: Path) -> None:
        draw_point_cloud(self.get_file_path(self.f8_调整地面在下, output_path.stem), output_path)

    @mark_as_method
    @mark_as_single_thread
    @mark_as_ply
    def f10_精细化对正(self, output_path: Path) -> None:
        cloud = read_point_cloud(self.get_input_path(self.f8_调整地面在下, output_path))
        points = np.asarray(cloud.points)
        best_rotation = least_squares_adjustment_direction(points)
        rotated_points = points.dot(best_rotation.T)
        cloud.points = Vector3dVector(rotated_points)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f11_绘制点云(self, output_path: Path) -> None:
        draw_point_cloud(self.get_file_path(self.f10_精细化对正, output_path), output_path)

    @mark_as_method
    @mark_as_ply
    @mark_as_single_thread
    def f12_仅保留顶面(self, output_path: Path) -> None:
        cloud = read_point_cloud(self.get_input_path(self.f10_精细化对正, output_path))
        points = np.asarray(cloud.points)
        point_z = points[:, 2]
        bottom_center = np.min(point_z)
        top_center = np.max(point_z)
        range_z = top_center - bottom_center
        z_selector = (point_z > (bottom_center + range_z * 0.1)) & (point_z < (top_center - range_z * 0.4))
        boundary_points = points[z_selector]
        point_x, point_y = boundary_points[:, 0], boundary_points[:, 1]
        thresholds = [0.1, 0.05, 0.03, 0.02, 0.01]
        left_center, right_center, front_center, back_center = find_valid_clusters(point_x, point_y, thresholds)
        self.print_safe(f'{left_center=} {right_center=}')
        self.print_safe(f'{front_center=} {back_center=}')
        assert back_center > front_center and right_center > left_center
        (extend_x, extend_y,
         definite_left, definite_right,
         definite_front, definite_back) = compute_extended_bounds(
            left_center, right_center,
            front_center, back_center
        )
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
                & (point_z > bottom_center + range_z * 0.5)
        )
        cloud.points = Vector3dVector(points[top_selector])
        colors = np.asarray(cloud.colors)
        if colors.size:
            cloud.colors = Vector3dVector(colors[top_selector])
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f13_绘制点云(self, output_path: Path) -> None:
        draw_point_cloud(self.get_file_path(self.f12_仅保留顶面, output_path.stem), output_path)

    @mark_as_method
    @mark_as_npy
    def f14_表面二维重建(self, output_path: Path) -> None:
        interpolated_matrix = surface_interpolate_2d(
            read_point_cloud(self.get_input_path(self.f12_仅保留顶面, output_path)), 0.1, 'cubic')
        for i, name in enumerate(['z', 'r', 'g', 'b'][:interpolated_matrix.shape[2]]):
            layer = interpolated_matrix[..., i]
            total, nan = layer.size, np.isnan(layer).sum()
            self.print_safe(f"Layer '{name}': {total=} {nan=} {nan / total * 100:.2f}%")
        np.save(output_path, interpolated_matrix)

    @mark_as_method
    def f15_绘制高程(self, output_path: Path) -> None:
        elevation = self.get_input_array(self.f14_表面二维重建, output_path)[..., 0]
        norm = plt.Normalize(*np.nanquantile(elevation, [0.01, 0.99]))
        Image.fromarray(
            (cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(elevation)[..., :3] * 255).astype(np.uint8)).save(
            output_path)

    @mark_as_method
    def f16_绘制图像(self, output_path: Path) -> None:
        if (matrix := self.get_input_array(self.f14_表面二维重建, output_path)).shape[2] > 1:
            color = np.clip([(matrix[..., i] - np.nanquantile(matrix[..., i], 0.01)) / (
                    np.nanquantile(matrix[..., i], 0.99) - np.nanquantile(matrix[..., i], 0.01) + 1e-9) * 255 for i
                             in range(1, 4)], 0, 255)
            Image.fromarray(np.round(color).astype(np.uint8).transpose(1, 2, 0)).save(output_path)

    @mark_as_method
    def f17_合并两张图(self, output_path: Path) -> None:
        elev_img = self.get_input_image(self.f15_绘制高程, output_path)
        surf_img = self.get_input_image(self.f16_绘制图像, output_path)
        if surf_img and (elev_img.size == surf_img.size):
            (Image.new('RGB', (elev_img.width + surf_img.width, elev_img.height))
             .paste(elev_img, (0, 0))
             .paste(surf_img, (elev_img.width, 0)).save(output_path))
        else:
            elev_img.save(output_path)


if __name__ == '__main__':
    s25022602_劈裂面形貌扫描_花岗岩_低曝光度.main()
