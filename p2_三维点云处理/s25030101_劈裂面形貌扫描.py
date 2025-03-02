import logging
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector

from batch_processor.batch_processor import SerialProcess, ManuallyProcessRequiredException
from batch_processor.processors.combined_excel import mark_as_combined_excel
from batch_processor.processors.npy import mark_as_npy
from batch_processor.processors.pickle import mark_as_pickle
from batch_processor.processors.ply import mark_as_ply
from batch_processor.processors.png import mark_as_png
from rock_texture_analyzer.point_clode.boundary_processing import get_boundaries
from rock_texture_analyzer.point_clode.interpolation import surface_interpolate_2d
from rock_texture_analyzer.point_clode.optimization import least_squares_adjustment_direction
from rock_texture_analyzer.point_clode.other_utils import should_flip_based_on_z, compute_rotation_matrix, point_cloud_keep_top, \
    depth_matrix_to_rgb_image, merge_5_images, depth_matrix_to_elevation_image


logger = logging.getLogger(Path(__name__).stem)


class s25030101_劈裂面形貌扫描(SerialProcess):
    is_debug = False

    @mark_as_ply
    def f1_原始数据(self, path: Path):
        raise ManuallyProcessRequiredException

    @mark_as_ply
    def f0201_读取点云原始数据(self, path: Path):
        return self.f1_原始数据.read(path)

    @mark_as_png
    def f0202_绘制点云(self, path: Path):
        return self.f0201_读取点云原始数据.read(path)

    @mark_as_ply
    def f0301_调整为主平面(self, path: Path):
        cloud = self.f0201_读取点云原始数据.read(path)
        points = np.asarray(cloud.points)
        points = points - np.mean(points, axis=0)
        plane_normal = np.linalg.svd(np.cov(points.T))[2][-1]
        plane_normal /= np.linalg.norm(plane_normal)
        R = compute_rotation_matrix(plane_normal, [0, 0, 1])
        cloud.points = Vector3dVector(np.dot(points, R.T))
        return cloud

    @mark_as_png
    def f0302_绘制点云(self, path: Path):
        return self.f0301_调整为主平面.read(path)

    @mark_as_ply
    def f0401_xOy平面对正(self, path: Path):
        cloud = self.f0301_调整为主平面.read(path)
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
            R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            cloud.points = Vector3dVector(np.dot(points, R.T))
        return cloud

    @mark_as_png
    def f0402_绘制点云(self, path: Path):
        return self.f0401_xOy平面对正.read(path)

    @mark_as_ply
    def f0501_调整地面在下(self, path: Path):
        cloud = self.f0401_xOy平面对正.read(path)
        points = np.asarray(cloud.points)
        if should_flip_based_on_z(points):
            cloud.points = Vector3dVector(-points)
        return cloud

    @mark_as_png
    def f0502_绘制点云(self, path: Path):
        return self.f0501_调整地面在下.read(path)

    @mark_as_ply
    def f0601_精细化对正(self, path: Path):
        cloud = self.f0501_调整地面在下.read(path)
        points = np.asarray(cloud.points)
        best_rotation = least_squares_adjustment_direction(points)
        rotated_points = points.dot(best_rotation.T)
        cloud.points = Vector3dVector(rotated_points)
        return cloud

    @mark_as_png
    def f0602_绘制点云(self, path: Path):
        return self.f0601_精细化对正.read(path)

    @mark_as_png
    def f0701_计算顶面与底面位置的KDE图(self, path: Path):
        figure: plt.Figure = plt.figure()
        cloud = self.f0601_精细化对正.read(path)
        z = np.asarray(cloud.points)[:, 2]
        ax = figure.subplots()
        sns.kdeplot(z, fill=True, ax=ax)
        return figure

    @mark_as_pickle
    def f0702_各个面的坐标(self, path: Path):
        cloud = self.f0601_精细化对正.read(path)
        points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
        x0, x1, y0, y1, z0, z1 = get_boundaries(points)
        logger.info(f'{x0=} {x1=} {y0=} {y1=} {z0=} {z1=}')
        return x0, x1, y0, y1, z0, z1

    @mark_as_ply
    def f0801_仅保留顶面(self, path: Path):
        cloud = self.f0601_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f0702_各个面的坐标.read(path)
        return point_cloud_keep_top(cloud, x0, x1, y0, y1, z0, z1)

    @mark_as_ply
    def f0802_仅保留左侧面(self, path: Path):
        cloud = self.f0601_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f0702_各个面的坐标.read(path)
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
        cloud.points = Vector3dVector(np.dot(np.asarray(cloud.points), R.T))
        return point_cloud_keep_top(cloud, z0, z1, y0, y1, x1, x0)

    @mark_as_ply
    def f0803_仅保留右侧面(self, path: Path):
        cloud = self.f0601_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f0702_各个面的坐标.read(path)
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
        cloud.points = Vector3dVector(np.dot(np.asarray(cloud.points), R.T))
        return point_cloud_keep_top(cloud, z0, z1, y0, y1, x0, x1)

    @mark_as_ply
    def f0804_仅保留前面(self, path: Path):
        cloud = self.f0601_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f0702_各个面的坐标.read(path)
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        cloud.points = Vector3dVector(np.dot(np.asarray(cloud.points), R.T))
        return point_cloud_keep_top(cloud, x0, x1, z0, z1, y1, y0)

    @mark_as_ply
    def f0805_仅保留后面(self, path: Path):
        cloud = self.f0601_精细化对正.read(path)
        x0, x1, y0, y1, z0, z1 = self.f0702_各个面的坐标.read(path)
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
        cloud.points = Vector3dVector(np.dot(np.asarray(cloud.points), R.T))
        return point_cloud_keep_top(cloud, x0, x1, z0, z1, y0, y1)

    @mark_as_png
    def f0901_绘制顶面点云(self, path: Path):
        return self.f0801_仅保留顶面.read(path)

    @mark_as_png
    def f0902_绘制左侧点云(self, path: Path):
        cloud = self.f0802_仅保留左侧面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.2, 'nearest')
        return depth_matrix_to_rgb_image(matrix)

    @mark_as_png
    def f0903_绘制右侧点云(self, path: Path):
        cloud = self.f0803_仅保留右侧面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.2, 'nearest')
        return depth_matrix_to_rgb_image(matrix)

    @mark_as_png
    def f0904_绘制前面点云(self, path: Path):
        cloud = self.f0804_仅保留前面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.2, 'nearest')
        return depth_matrix_to_rgb_image(matrix)

    @mark_as_png
    def f0905_绘制后面点云(self, path: Path):
        cloud = self.f0805_仅保留后面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.2, 'nearest')
        return depth_matrix_to_rgb_image(matrix)

    @mark_as_npy
    def f1001_表面二维重建(self, path: Path):
        cloud = self.f0801_仅保留顶面.read(path)
        matrix = surface_interpolate_2d(cloud, 0.1, 'cubic')
        for i, name in enumerate(['z', 'r', 'g', 'b'][:matrix.shape[2]]):
            layer = matrix[..., i]
            total, nan = layer.size, np.isnan(layer).sum()
            logger.info(f"Layer '{name}': {total=} {nan=} {nan / total * 100:.2f}%")
        return matrix

    @mark_as_png
    def f1002_绘制高程(self, path: Path):
        return depth_matrix_to_elevation_image(self.f1001_表面二维重建.read(path))

    @mark_as_png
    def f1003_绘制图像(self, path: Path):
        return depth_matrix_to_rgb_image(self.f1001_表面二维重建.read(path))

    @mark_as_png
    def f1004_合并两张图(self, path: Path):
        """将高程图与表面图合并为横向排列的图片"""
        elevation_img = self.f1002_绘制高程.read(path)
        surface_img = self.f1003_绘制图像.read(path)
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
    def f1101_合并全部的图(self, path: Path) -> Image.Image:
        """合并所有方向图像为单个图像"""
        return merge_5_images(
            center=self.f1003_绘制图像.read(path),
            left=self.f0902_绘制左侧点云.read(path),
            right=self.f0903_绘制右侧点云.read(path),
            front=self.f0904_绘制前面点云.read(path),
            back=self.f0905_绘制后面点云.read(path),
        )

    @mark_as_combined_excel(columns=('顺时针旋转次数',))
    def f1102_旋转与翻转方向(self, path: Path):
        return -1,

    @mark_as_png
    def f1103_按要求进行旋转(self, path: Path):
        v1, = self.f1102_旋转与翻转方向.read(path)
        if v1 == -1:
            raise ManuallyProcessRequiredException('需要调整表格里面的数据来指定旋转方向')
        return self.f1101_合并全部的图.read(path).rotate(-90 * v1)

    @mark_as_npy
    def f1104_旋转使得方向一致(self, path: Path):
        matrix = self.f1001_表面二维重建.read(path)
        v1, = self.f1102_旋转与翻转方向.read(path)
        return np.rot90(matrix, k=-v1, axes=(0, 1))

    @mark_as_png
    def f1105_高程图(self, path: Path):
        return depth_matrix_to_elevation_image(self.f1104_旋转使得方向一致.read(path))

    @mark_as_png
    def f1106_色彩图(self, path: Path):
        return depth_matrix_to_rgb_image(self.f1104_旋转使得方向一致.read(path))

    @mark_as_npy
    def f1201_翻转使得上下面可以比较(self, path: Path):
        need_invert = path.stem[-2] in {'U', 'u'}
        matrix = self.f1104_旋转使得方向一致.read(path)
        if need_invert:
            matrix = matrix[:, ::-1, :]
            matrix[..., 0] *= -1
        return matrix

    @mark_as_png
    def f1202_高程图(self, path: Path):
        return depth_matrix_to_elevation_image(self.f1201_翻转使得上下面可以比较.read(path))

    @mark_as_png
    def f1203_色彩图(self, path: Path):
        return depth_matrix_to_rgb_image(self.f1201_翻转使得上下面可以比较.read(path))


if __name__ == '__main__':
    s25030101_劈裂面形貌扫描.main()
