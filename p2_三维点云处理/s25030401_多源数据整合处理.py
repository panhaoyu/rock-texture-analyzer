import logging
from pathlib import Path

import numpy as np

from batch_processor import SerialProcess, mark_as_npy, mark_as_png, mark_as_source
from rock_texture_analyzer.image_4d.fix_nan import remove_nan_borders, fill_nan_values
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


if __name__ == '__main__':
    s25030401_多源数据整合处理.main()
