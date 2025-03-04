import logging
from pathlib import Path

import numpy as np

from batch_processor import SerialProcess, mark_as_npy, mark_as_png, mark_as_source, mark_as_recreate
from rock_texture_analyzer.image_4d.fix_nan import remove_nan_borders, fill_nan_values
from rock_texture_analyzer.image_4d.plotting import merge_image_grid, depth_matrix_to_elevation_image, add_label
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
    @mark_as_png
    def f0501_展示目前已有数据(self):
        return merge_image_grid([[
            add_label(depth_matrix_to_elevation_image(self.f0102_剪切前_劈裂面_三维扫描_上半部分, 10), '剪切前 上'),
            add_label(self.f0104_剪切前_劈裂面_光学扫描_上半部分.resize((1000, 1000)), '剪切前 上'),
            add_label(self.f0101_剪切前_侧面_光学扫描.resize((1000, 1000)), '剪切前 侧'),
        ], [
            add_label(depth_matrix_to_elevation_image(self.f0103_剪切前_劈裂面_三维扫描_下半部分, 10), '剪切前 下'),
            add_label(self.f0105_剪切前_劈裂面_光学扫描_下半部分.resize((1000, 1000)), '剪切前 下'),
            None,
        ], [
            add_label(depth_matrix_to_elevation_image(self.f0201_剪切前_劈裂面_高程差值, 1), '剪切前 差值'),
            None,
            None,
        ]])


if __name__ == '__main__':
    s25030401_多源数据整合处理.main()
