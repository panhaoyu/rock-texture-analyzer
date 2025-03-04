import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageCms

from batch_processor import SerialProcess, mark_as_npy, mark_as_png, mark_as_source, mark_as_recreate
from rock_texture_analyzer.image_4d.fix_nan import remove_nan_borders, fill_nan_values
from rock_texture_analyzer.image_4d.plotting import merge_image_grid, matrix_to_elevation_image, add_label, \
    matrix_to_rgb_image
from rock_texture_analyzer.image_4d.scaling import scale_array

logger = logging.getLogger(Path(__file__).name)


def process(array: np.ndarray) -> np.ndarray:
    array = remove_nan_borders(array)
    array = fill_nan_values(array)
    array = scale_array(array)
    return array


class s25030401_多源数据整合处理(SerialProcess):
    @mark_as_source
    @mark_as_png
    def f0101_剪切前_侧面_光学扫描(self):
        pass

    @mark_as_source
    @mark_as_npy
    def f0102_剪切前_劈裂面_三维扫描_上半部分(self):
        pass

    @mark_as_source
    @mark_as_npy
    def f0103_剪切前_劈裂面_三维扫描_下半部分(self):
        pass

    @mark_as_source
    @mark_as_png
    def f0104_剪切前_劈裂面_光学扫描_上半部分(self):
        pass

    @mark_as_source
    @mark_as_png
    def f0105_剪切前_劈裂面_光学扫描_下半部分(self):
        pass

    @mark_as_npy
    def f0201_剪切前_劈裂面_高程差值(self):
        v1, v2 = self.f0102_剪切前_劈裂面_三维扫描_上半部分, self.f0103_剪切前_劈裂面_三维扫描_下半部分
        return v1 - v2

    @mark_as_recreate
    @mark_as_npy
    def f0202_剪切前_劈裂面_光学差值(self):
        img1, img2 = self.f0104_剪切前_劈裂面_光学扫描_上半部分, self.f0105_剪切前_劈裂面_光学扫描_下半部分
        # 转换为LAB颜色空间并计算色差
        srgb_p, lab_p = ImageCms.createProfile('sRGB'), ImageCms.createProfile('LAB')
        transform = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, 'RGB', 'LAB')
        # 执行颜色空间转换
        lab1 = ImageCms.applyTransform(img1, transform)
        lab2 = ImageCms.applyTransform(img2, transform)
        lab1 = np.asarray(lab1)
        lab2 = np.asarray(lab2)

        delta = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))

        # 应用容差阈值并执行形态学操作
        delta = (delta > 20).astype(np.uint8)
        kernel_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        delta = cv2.dilate(delta, kernel)  # 先膨胀连接断裂区域
        delta = cv2.erode(delta, kernel)  # 再腐蚀消除孤立噪点

        return delta

    @mark_as_recreate
    @mark_as_png
    def f0202_剪切前_劈裂面_光学差值_绘图(self):
        array = self.f0202_剪切前_劈裂面_光学差值
        array = np.array([[0, 0, 0], [255, 255, 255]])[array]
        array = array.astype(np.uint8)
        return array

    @mark_as_recreate
    @mark_as_png
    def f0501_展示目前已有数据(self):
        return merge_image_grid([[
            add_label(matrix_to_elevation_image(self.f0102_剪切前_劈裂面_三维扫描_上半部分, 10), '剪切前 上 高程'),
            add_label(matrix_to_rgb_image(self.f0102_剪切前_劈裂面_三维扫描_上半部分), '剪切前 上 纹理'),
            add_label(self.f0104_剪切前_劈裂面_光学扫描_上半部分.resize((1000, 1000)), '剪切前 上 光学'),
            add_label(self.f0101_剪切前_侧面_光学扫描.resize((1000, 1000)), '剪切前 侧 光学'),
        ], [
            add_label(matrix_to_elevation_image(self.f0103_剪切前_劈裂面_三维扫描_下半部分, 10), '剪切前 下 高程'),
            add_label(matrix_to_rgb_image(self.f0103_剪切前_劈裂面_三维扫描_下半部分), '剪切前 下 纹理'),
            add_label(self.f0105_剪切前_劈裂面_光学扫描_下半部分.resize((1000, 1000)), '剪切前 下 光学'),
        ], [
            add_label(matrix_to_elevation_image(self.f0201_剪切前_劈裂面_高程差值, 2), '剪切前 差值 高程'),
            None,
            add_label(self.f0202_剪切前_劈裂面_光学差值_绘图.resize((1000, 1000)), '剪切前 差值 光学'),
        ]])


if __name__ == '__main__':
    s25030401_多源数据整合处理.main()
