import logging
from pathlib import Path

import numpy as np
from PIL import Image

from batch_processor import BatchProcessor, mark_as_npy, mark_as_png
from batch_processor.processors.base import ManuallyProcessRequiredException
from rock_texture_analyzer.image_4d.fix_nan import remove_nan_borders, fill_nan_values
from rock_texture_analyzer.image_4d.scaling import scale_array
from rock_texture_analyzer.point_clode.other_utils import depth_matrix_to_elevation_image, depth_matrix_to_rgb_image

logger = logging.getLogger(Path(__file__).name)


def process(array: np.ndarray) -> np.ndarray:
    array = remove_nan_borders(array)
    array = fill_nan_values(array)
    array = scale_array(array)
    return array


class s25030102_劈裂面形貌扫描对比(BatchProcessor):
    @mark_as_npy
    def f0101_原始数据_Da(self, path: Path):
        raise ManuallyProcessRequiredException

    @mark_as_npy
    def f0102_原始数据_Db(self, path: Path):
        raise ManuallyProcessRequiredException

    @mark_as_npy
    def f0103_原始数据_Ua(self, path: Path):
        raise ManuallyProcessRequiredException

    @mark_as_npy
    def f0104_原始数据_Ub(self, path: Path):
        raise ManuallyProcessRequiredException

    @mark_as_npy
    def f0201_DA放缩(self, path: Path):
        array = self.f0101_原始数据_Da.read(path)
        return process(array)

    @mark_as_npy
    def f0202_DB放縮(self, path: Path):
        array = self.f0102_原始数据_Db.read(path)
        return process(array)

    @mark_as_npy
    def f0203_UA放縮(self, path: Path):
        array = self.f0103_原始数据_Ua.read(path)
        return process(array)

    @mark_as_npy
    def f0204_UB放縮(self, path: Path):
        array = self.f0104_原始数据_Ub.read(path)
        return process(array)

    @mark_as_png
    def f0301_合并显示(self, path: Path):
        elevation_ua = depth_matrix_to_elevation_image(self.f0203_UA放縮.read(path))
        colorful_ua = depth_matrix_to_rgb_image(self.f0203_UA放縮.read(path))
        elevation_ub = depth_matrix_to_elevation_image(self.f0204_UB放縮.read(path))
        colorful_ub = depth_matrix_to_rgb_image(self.f0204_UB放縮.read(path))
        elevation_da = depth_matrix_to_elevation_image(self.f0201_DA放缩.read(path))
        colorful_da = depth_matrix_to_rgb_image(self.f0201_DA放缩.read(path))
        elevation_db = depth_matrix_to_elevation_image(self.f0202_DB放縮.read(path))
        colorful_db = depth_matrix_to_rgb_image(self.f0202_DB放縮.read(path))

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
