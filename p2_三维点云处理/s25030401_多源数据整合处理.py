import logging
from pathlib import Path

import numpy as np

from batch_processor import SerialProcess, mark_as_npy, mark_as_png, mark_as_source, mark_as_recreate
from rock_texture_analyzer.image_4d.fix_nan import remove_nan_borders, fill_nan_values
from rock_texture_analyzer.image_4d.plotting import merge_image_grid, matrix_to_elevation_image, add_label, \
    matrix_to_rgb_image
from rock_texture_analyzer.image_4d.scaling import scale_array
from rock_texture_analyzer.image_process.convert import boolean_to_image
from rock_texture_analyzer.image_process.image_diff import image_diff

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
        diff = image_diff(img1, img2, threshold=15, kernel_size=3)
        return diff

    @mark_as_recreate
    @mark_as_png
    def f0202_剪切前_劈裂面_光学差值_绘图(self):
        return boolean_to_image(self.f0202_剪切前_劈裂面_光学差值)

    @mark_as_png
    def f0204_剪切前_劈裂面_光学差值_参数比较(self):
        img1, img2 = self.f0104_剪切前_劈裂面_光学扫描_上半部分, self.f0105_剪切前_劈裂面_光学扫描_下半部分
        result = []
        for kernel_size in (2, 3, 5, 10, 20):
            row = []
            result.append(row)
            for delta in (5, 10, 14, 18, 22, 26, 30):
                diff = image_diff(img1, img2, threshold=delta, kernel_size=kernel_size)
                diff = boolean_to_image(diff)
                row.append(diff)
        return merge_image_grid(result)

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
