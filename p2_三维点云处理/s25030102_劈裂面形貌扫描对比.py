import logging
from pathlib import Path

import numpy as np
from PIL import Image

from batch_processor import SerialProcess, mark_as_npy, mark_as_png
from batch_processor.batch_processor import ManuallyProcessRequiredException
from rock_texture_analyzer.image_4d.fix_nan import remove_nan_borders, fill_nan_values
from rock_texture_analyzer.image_4d.scaling import scale_array
from rock_texture_analyzer.point_clode.other_utils import depth_matrix_to_elevation_image, depth_matrix_to_rgb_image

logger = logging.getLogger(Path(__file__).name)


def process(array: np.ndarray) -> np.ndarray:
    array = remove_nan_borders(array)
    array = fill_nan_values(array)
    array = scale_array(array)
    return array


class s25030102_劈裂面形貌扫描对比(SerialProcess):
    @mark_as_npy
    def f0101_原始数据_Da(self):
        raise ManuallyProcessRequiredException

    @mark_as_npy
    def f0102_原始数据_Db(self):
        raise ManuallyProcessRequiredException

    @mark_as_npy
    def f0103_原始数据_Ua(self):
        raise ManuallyProcessRequiredException

    @mark_as_npy
    def f0104_原始数据_Ub(self):
        raise ManuallyProcessRequiredException

    @mark_as_npy
    def f0201_DA放缩(self):
        array = self.f0101_原始数据_Da
        return process(array)

    @mark_as_npy
    def f0202_DB放縮(self):
        array = self.f0102_原始数据_Db
        return process(array)

    @mark_as_npy
    def f0203_UA放縮(self):
        array = self.f0103_原始数据_Ua
        return process(array)

    @mark_as_npy
    def f0204_UB放縮(self):
        array = self.f0104_原始数据_Ub
        return process(array)

    @mark_as_png
    def f0301_合并显示(self):
        elevation_ua = depth_matrix_to_elevation_image(self.f0203_UA放縮)
        colorful_ua = depth_matrix_to_rgb_image(self.f0203_UA放縮)
        elevation_ub = depth_matrix_to_elevation_image(self.f0204_UB放縮)
        colorful_ub = depth_matrix_to_rgb_image(self.f0204_UB放縮)
        elevation_da = depth_matrix_to_elevation_image(self.f0201_DA放缩)
        colorful_da = depth_matrix_to_rgb_image(self.f0201_DA放缩)
        elevation_db = depth_matrix_to_elevation_image(self.f0202_DB放縮)
        colorful_db = depth_matrix_to_rgb_image(self.f0202_DB放縮)

        # 组合图像矩阵
        upper_row = [elevation_ua, elevation_ub, colorful_ua, colorful_ub]
        lower_row = [elevation_da, elevation_db, colorful_da, colorful_db]

        w, h = upper_row[0].size
        merged = Image.new('RGB', (w * 4, h * 2))
        [merged.paste(img, (i * w, 0)) for i, img in enumerate(upper_row)]
        [merged.paste(img, (i * w, h)) for i, img in enumerate(lower_row)]
        return merged


if __name__ == '__main__':
    s25030102_劈裂面形貌扫描对比.main()
