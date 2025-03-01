import logging
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import cm, pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector

from batch_processor.batch_processor import BatchProcessor
from batch_processor.processors.base import ManuallyProcessRequiredException, mark_as_single_thread
from batch_processor.processors.combined_excel import mark_as_combined_excel
from batch_processor.processors.npy import mark_as_npy
from batch_processor.processors.pickle import mark_as_pickle
from batch_processor.processors.ply import mark_as_ply
from batch_processor.processors.png import mark_as_png
from rock_texture_analyzer.boundary_processing import get_boundaries
from rock_texture_analyzer.interpolation import surface_interpolate_2d
from rock_texture_analyzer.optimization import least_squares_adjustment_direction
from rock_texture_analyzer.other_utils import should_flip_based_on_z, compute_rotation_matrix, point_cloud_keep_top, \
    point_cloud_top_projection, merge_5_images

logging.basicConfig(
    level=logging.INFO,
    style='{',
    datefmt='%H%M%S',
    format="{levelname:>8} {asctime} {name:<20} {message}"
)


class s25022602_劈裂面形貌扫描_花岗岩_低曝光度(BatchProcessor):
    is_debug = False

    @mark_as_ply
    def f1_原始数据(self, path: Path):
        raise ManuallyProcessRequiredException

    @mark_as_ply
    def f2_读取点云原始数据(self, path: Path):
        return self.f1_原始数据.read(path)

    @mark_as_single_thread
    @mark_as_png
    def f3_绘制点云(self, path: Path):
        return self.f2_读取点云原始数据.read(path)

    @mark_as_ply
    def f4_调整为主平面(self, path: Path):
        cloud = self.f2_读取点云原始数据.read(path)
        points = np.asarray(cloud.points)
        centered_points = points - np.mean(points, axis=0)
        plane_normal = np.linalg.svd(np.cov(centered_points.T))[2][-1]
        plane_normal /= np.linalg.norm(plane_normal)
        rotation_matrix = compute_rotation_matrix(plane_normal, [0, 0, 1])
        cloud.points = Vector3dVector(centered_points @ rotation_matrix.T)
        return cloud

    @mark_as_single_thread
    @mark_as_png
    def f5_绘制点云(self, path: Path):
        return self.f4_调整为主平面.read(path)

    @mark_as_ply
    def f6_xOy平面对正(self, path: Path):
        cloud = self.f4_调整为主平面.read(path)
        points = np.asarray(cloud.points)
        x, y = points[:, :2].T
        x_bins = np.arange(x.min(), x.max() + 1)
        y_bins = np.arange(y.min(), y.max() + 1)
        hist = np.histogram2d(x, y, bins=[x_bins, y_bins])[0]
        density_image = np.where(hist > 50, 255, 0).astype(np.uint8)[::-1]
        if contours := cv2.findContours(density_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))[-1]
            angle = angle if angle < -45 else angle + 90
            theta = np.radians(-angle)
            R_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            cloud.points = Vector3dVector(points @ R_z.T)
        return cloud

    @mark_as_single_thread
    @mark_as_png
    def f7_绘制点云(self, path: Path):
        return self.f6_xOy平面对正.read(path)

    @mark_as_single_thread
    @mark_as_ply
    def f8_调整地面在下(self, path: Path):
        cloud = self.f6_xOy平面对正.read(path)
        points = np.asarray(cloud.points)
        if should_flip_based_on_z(points):
            cloud.points = Vector3dVector(-points)
        return cloud

    @mark_as_single_thread
    @mark_as_png
    def f9_绘制点云(self, path: Path):
        return self.f8_调整地面在下.read(path)

    @mark_as_single_thread
    @mark_as_ply
    def f10_精细化对正(self, path: Path):
        cloud = self.f8_调整地面在下.read(path)
        points = np.asarray(cloud.points)
        best_rotation = least_squares_adjustment_direction(points)
        rotated_points = points.dot(best_rotation.T)
        cloud.points = Vector3dVector(rotated_points)
        return cloud

    @mark_as_single_thread
    @mark_as_png
    def f11_绘制点云(self, path: Path):
        return self.f10_精细化对正.read(path)

    @mark_as_png
    def f11_计算顶面与底面位置的KDE图(self, path: Path):
        figure: plt.Figure = plt.figure()
        cloud = self.f10_精细化对正.read(path)
        z = np.asarray(cloud.points)[:, 2]
        ax = figure.subplots()
        sns.kdeplot(z, fill=True, ax=ax)
        return figure

    @mark_as_pickle
    def f12_各个面的坐标(self, path: Path):
        cloud = self.f10_精细化对正.read(path)
        points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
        x0, x1, y0, y1, z0, z1 = get_boundaries(points)
        logger.info(f'{x0=} {x1=} {y0=} {y1=} {z0=} {z1=}')
        return x0, x1, y0, y1, z0, z1

    @mark_as_single_thread
    @mark_as_ply
    def f13_1_仅保留顶面(self, path: Path):
        cloud = self.f10_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f12_各个面的坐标.read(path)
        return point_cloud_keep_top(cloud, x0, x1, y0, y1, z0, z1)

    @mark_as_single_thread
    @mark_as_ply
    def f13_2_仅保留左侧面(self, path: Path):
        cloud = self.f10_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f12_各个面的坐标.read(path)
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
        cloud.points = Vector3dVector(np.asarray(cloud.points) @ R)
        return point_cloud_keep_top(cloud, z0, z1, y0, y1, x1, x0)

    @mark_as_single_thread
    @mark_as_ply
    def f13_3_仅保留右侧面(self, path: Path):
        cloud = self.f10_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f12_各个面的坐标.read(path)
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
        cloud.points = Vector3dVector(np.asarray(cloud.points) @ R)
        return point_cloud_keep_top(cloud, z0, z1, y0, y1, x0, x1)

    @mark_as_single_thread
    @mark_as_ply
    def f13_4_仅保留前面(self, path: Path):
        cloud = self.f10_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f12_各个面的坐标.read(path)
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
        cloud.points = Vector3dVector(np.asarray(cloud.points) @ R)
        return point_cloud_keep_top(cloud, x0, x1, z0, z1, y1, y0)

    @mark_as_single_thread
    @mark_as_ply
    def f13_5_仅保留后面(self, path: Path):
        cloud = self.f10_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f12_各个面的坐标.read(path)
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        cloud.points = Vector3dVector(np.asarray(cloud.points) @ R)
        return point_cloud_keep_top(cloud, x0, x1, z0, z1, y0, y1)

    @mark_as_png
    def f14_1_绘制顶面点云(self, path: Path):
        return self.f13_1_仅保留顶面.read(path)

    @mark_as_png
    def f14_2_绘制左侧点云(self, path: Path):
        cloud = self.f13_2_仅保留左侧面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.2, 'nearest')
        return point_cloud_top_projection(matrix)

    @mark_as_png
    def f14_3_绘制右侧点云(self, path: Path):
        cloud = self.f13_3_仅保留右侧面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.2, 'nearest')
        return point_cloud_top_projection(matrix)

    @mark_as_png
    def f14_4_绘制前面点云(self, path: Path):
        cloud = self.f13_4_仅保留前面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.2, 'nearest')
        return point_cloud_top_projection(matrix)

    @mark_as_png
    def f14_5_绘制后面点云(self, path: Path):
        cloud = self.f13_5_仅保留后面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.2, 'nearest')
        return point_cloud_top_projection(matrix)

    @mark_as_npy
    def f14_表面二维重建(self, path: Path):
        cloud = self.f13_1_仅保留顶面.read(path)
        interpolated_matrix = surface_interpolate_2d(cloud, 0.1, 'cubic')
        for i, name in enumerate(['z', 'r', 'g', 'b'][:interpolated_matrix.shape[2]]):
            layer = interpolated_matrix[..., i]
            total, nan = layer.size, np.isnan(layer).sum()
            logger.info(f"Layer '{name}': {total=} {nan=} {nan / total * 100:.2f}%")
        return interpolated_matrix

    @mark_as_png
    def f15_绘制高程(self, path: Path):
        elevation = self.f14_表面二维重建.read(path)[..., 0]
        norm = plt.Normalize(*np.nanquantile(elevation, [0.01, 0.99]))
        return (cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(elevation)[..., :3] * 255).astype(np.uint8)

    @mark_as_png
    def f16_绘制图像(self, path: Path) -> np.ndarray:
        matrix = self.f14_表面二维重建.read(path)
        return point_cloud_top_projection(matrix)

    @mark_as_png
    def f17_合并两张图(self, path: Path):
        """将高程图与表面图合并为横向排列的图片"""
        elevation_img = self.f15_绘制高程.read(path)
        surface_img = self.f16_绘制图像.read(path)
        if not surface_img:
            elevation_img.save(path)
            return
        if (size := elevation_img.size) != surface_img.size:
            raise ValueError("高程图与表面图尺寸不一致")
        combined_img = Image.new('RGB', (size[0] * 2, size[1]))
        combined_img.paste(elevation_img, (0, 0))
        combined_img.paste(surface_img, (size[0], 0))
        return combined_img

    @mark_as_png
    def f18_合并全部的图(self, path: Path) -> Image.Image:
        """合并所有方向图像为单个图像"""
        return merge_5_images(
            center=self.f16_绘制图像.read(path),
            left=self.f14_2_绘制左侧点云.read(path),
            right=self.f14_3_绘制右侧点云.read(path),
            front=self.f14_4_绘制前面点云.read(path),
            back=self.f14_5_绘制后面点云.read(path),
        )

    @mark_as_combined_excel(columns=('顺时针旋转次数', '翻转'))
    def f19_旋转与翻转方向(self, path: Path):
        return -1, -1

    @mark_as_png
    def f20_按要求进行旋转与翻转(self, path: Path):
        v1, v2 = self.f19_旋转与翻转方向.read(path)
        if v1 == -1 or v2 == -1:
            raise ManuallyProcessRequiredException('Open the previous excel to specify the rotation and flip params')


if __name__ == '__main__':
    s25022602_劈裂面形貌扫描_花岗岩_低曝光度.main()
