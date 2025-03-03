import itertools
import logging
from pathlib import Path

import numpy as np
import seaborn
from PIL import Image
from matplotlib import pyplot as plt

from batch_processor import SerialProcess, mark_as_npy, mark_as_png, mark_as_source
from rock_texture_analyzer.image_4d.fix_nan import remove_nan_borders, fill_nan_values
from rock_texture_analyzer.image_4d.plotting import depth_matrix_to_rgb_image, \
    depth_matrix_to_elevation_image, merge_image_grid
from rock_texture_analyzer.image_4d.scaling import scale_array
from rock_texture_analyzer.image_4d.tilt_correction import tilt_correction

logger = logging.getLogger(Path(__file__).name)


def process(array: np.ndarray) -> np.ndarray:
    array = remove_nan_borders(array)
    array = fill_nan_values(array)
    array = scale_array(array)
    return array


class s25030102_多源数据整合处理(SerialProcess):
    @mark_as_source
    @mark_as_npy
    def f0101_剪切前_劈裂面_三维扫描_上半部分_第1次扫描(self):
        pass

    @mark_as_source
    @mark_as_npy
    def f0102_剪切前_劈裂面_三维扫描_上半部分_第2次扫描(self):
        pass

    @mark_as_source
    @mark_as_npy
    def f0103_剪切前_劈裂面_三维扫描_下半部分_第1次扫描(self):
        pass

    @mark_as_source
    @mark_as_npy
    def f0104_剪切前_劈裂面_三维扫描_下半部分_第2次扫描(self):
        pass

    @mark_as_source
    @mark_as_png
    def f0105_剪切前_劈裂面_光学扫描_上半部分(self):
        pass

    @mark_as_source
    @mark_as_png
    def f0106_剪切前_劈裂面_光学扫描_下半部分(self):
        pass

    @mark_as_source
    @mark_as_png
    def f0107_侧面光学扫描(self):
        pass

    @mark_as_npy
    def f0201_DA放缩(self):
        array = self.f0103_剪切前_劈裂面_三维扫描_下半部分_第1次扫描
        return process(array)

    @mark_as_npy
    def f0202_DB放缩(self):
        array = self.f0104_剪切前_劈裂面_三维扫描_下半部分_第2次扫描
        return process(array)

    @mark_as_npy
    def f0203_UA放缩(self):
        array = self.f0101_剪切前_劈裂面_三维扫描_上半部分_第1次扫描
        return process(array)

    @mark_as_npy
    def f0204_UB放缩(self):
        array = self.f0102_剪切前_劈裂面_三维扫描_上半部分_第2次扫描
        return process(array)

    @mark_as_png
    def f0301_合并显示(self):
        elevation_ua = depth_matrix_to_elevation_image(self.f0203_UA放缩)
        colorful_ua = depth_matrix_to_rgb_image(self.f0203_UA放缩)
        elevation_ub = depth_matrix_to_elevation_image(self.f0204_UB放缩)
        colorful_ub = depth_matrix_to_rgb_image(self.f0204_UB放缩)
        elevation_da = depth_matrix_to_elevation_image(self.f0201_DA放缩)
        colorful_da = depth_matrix_to_rgb_image(self.f0201_DA放缩)
        elevation_db = depth_matrix_to_elevation_image(self.f0202_DB放缩)
        colorful_db = depth_matrix_to_rgb_image(self.f0202_DB放缩)

        # 组合图像矩阵
        upper_row = [elevation_ua, elevation_ub, colorful_ua, colorful_ub]
        lower_row = [elevation_da, elevation_db, colorful_da, colorful_db]

        w, h = upper_row[0].size
        merged = Image.new('RGB', (w * 4, h * 2))
        [merged.paste(img, (i * w, 0)) for i, img in enumerate(upper_row)]
        [merged.paste(img, (i * w, h)) for i, img in enumerate(lower_row)]
        return merged

    @mark_as_png
    def f0401_比较各个扫描结果的差异(self):
        arrays = self.f0203_UA放缩, self.f0204_UB放缩, self.f0201_DA放缩, self.f0202_DB放缩

        def get_compare(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
            arr1, arr2 = tilt_correction(arr1, arr2)
            arr1, arr2 = arr1[:, :, 0], arr2[:, :, 0]
            delta = arr1 - arr2
            return np.abs(delta - np.mean(delta))

        compare = [[get_compare(arr1, arr2) for arr2 in arrays] for arr1 in arrays]
        return merge_image_grid([[depth_matrix_to_elevation_image(col, v_range=2) for col in row] for row in compare])

    @mark_as_npy
    def f0402_有效扫描数据(self):
        # 取最匹配的两个扫描结果
        upper_arrays = self.f0203_UA放缩, self.f0204_UB放缩
        lower_arrays = self.f0201_DA放缩, self.f0202_DB放缩
        pairs = list(itertools.product(upper_arrays, lower_arrays))
        pairs = [tilt_correction(img1, img2) for img1, img2 in pairs]
        errors = [upper - lower for upper, lower in pairs]
        errors = [np.mean(np.abs(error - np.mean(error))) for error in errors]
        upper, lower = pairs[np.argmin(errors)]
        return np.dstack([upper, lower])

    @mark_as_npy
    def f0403_上表面数据(self):
        return self.f0402_有效扫描数据[:, :, 0:4]

    @mark_as_npy
    def f0404_下表面数据(self):
        return self.f0402_有效扫描数据[:, :, 4:8]

    @mark_as_png
    def f0405_绘图(self):
        v1, v2 = self.f0403_上表面数据, self.f0404_下表面数据
        ua, ub, da, db = self.f0203_UA放缩, self.f0204_UB放缩, self.f0201_DA放缩, self.f0202_DB放缩
        v_range = 10
        dv = v1 - v2

        selected = merge_image_grid([[
            depth_matrix_to_elevation_image(v1, v_range=v_range, text=f'上部 高程 {v_range}mm'),
            depth_matrix_to_rgb_image(v1, text='上部 纹理')
        ], [
            depth_matrix_to_elevation_image(v2, v_range=v_range, text=f'下部 高程 {v_range}mm'),
            depth_matrix_to_rgb_image(v2, text='下部 纹理')

        ]])
        elevation_image = merge_image_grid([[
            depth_matrix_to_elevation_image(ua, v_range=v_range, text=f'上部A 高程 {v_range}mm'),
            depth_matrix_to_elevation_image(ub, v_range=v_range, text=f'上部B 高程 {v_range}mm')
        ], [
            depth_matrix_to_elevation_image(da, v_range=v_range, text=f'下部A 高程 {v_range}mm'),
            depth_matrix_to_elevation_image(db, v_range=v_range, text=f'下部B 高程 {v_range}mm')
        ]])

        rgb_image = merge_image_grid([[
            depth_matrix_to_rgb_image(ua, text='上部A 纹理'),
            depth_matrix_to_rgb_image(ub, text='上部B 纹理')
        ], [
            depth_matrix_to_rgb_image(da, text='下部A 纹理'),
            depth_matrix_to_rgb_image(db, text='下部B 纹理')
        ]])
        error = merge_image_grid([[
            depth_matrix_to_elevation_image(dv, v_range=0.2, text="差值 高程 0.2mm"),
            depth_matrix_to_elevation_image(dv, v_range=0.5, text="差值 高程 0.5mm")
        ], [
            depth_matrix_to_elevation_image(dv, v_range=1, text="差值 高程 1mm"),
            depth_matrix_to_elevation_image(dv, v_range=2, text="差值 高程 2mm")
        ]])

        return merge_image_grid([
            [selected, elevation_image],
            [error, rgb_image]
        ])

    @mark_as_png
    def f0406_各个像素的误差分布(self):
        v1, v2 = self.f0403_上表面数据[:, :, 0], self.f0404_下表面数据[:, :, 0]
        delta = v1 - v2
        delta = delta - np.mean(delta)
        delta = delta.ravel()
        axes: plt.Axes
        figure, axes = plt.subplots(1, 1)
        seaborn.kdeplot(delta, ax=axes, clip=(-1.1, 1.1), fill=True)
        axes.set_xlim(-1, 1)
        return figure


if __name__ == '__main__':
    s25030102_多源数据整合处理.main()
