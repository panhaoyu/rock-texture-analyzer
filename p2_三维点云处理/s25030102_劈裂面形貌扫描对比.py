import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont

from batch_processor import SerialProcess, mark_as_npy, mark_as_png, mark_as_recreate
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
    def f0202_DB放缩(self):
        array = self.f0102_原始数据_Db
        return process(array)

    @mark_as_npy
    def f0203_UA放缩(self):
        array = self.f0103_原始数据_Ua
        return process(array)

    @mark_as_npy
    def f0204_UB放缩(self):
        array = self.f0104_原始数据_Ub
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
        # ua, ub, da, db
        arrays = self.f0203_UA放缩, self.f0204_UB放缩, self.f0201_DA放缩, self.f0202_DB放缩
        arrays = [array[:, :, 0] for array in arrays]
        compare = [[arr1 - arr2 for arr2 in arrays] for arr1 in arrays]
        compare = [[np.abs(v - np.mean(v)) for v in vs] for vs in compare]
        array = np.hstack([np.vstack(v) for v in compare])
        figure, axes = plt.subplots(1, 1, figsize=(12, 10))
        im = axes.imshow(array, vmin=0, vmax=10, cmap='jet')
        figure.colorbar(im)
        return figure

    @mark_as_npy
    def f0402_有效扫描数据(self):
        # 取最匹配的两个扫描结果
        upper_arrays = self.f0203_UA放缩, self.f0204_UB放缩
        lower_arrays = self.f0201_DA放缩, self.f0202_DB放缩
        pairs = list(itertools.product(upper_arrays, lower_arrays))
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

    @mark_as_recreate
    @mark_as_png
    def f0405_绘图(self):
        def 添加文本(image: Image.Image, text: str) -> Image.Image:
            drawer = ImageDraw.Draw(image)
            font = ImageFont.truetype("Times New Roman.ttf", 50)
            textbox = drawer.textbbox((0, 0), text, font=font)
            drawer.rectangle((5, 5, textbox[2] - textbox[0] + 15, textbox[3] - textbox[1] + 15), fill="white")
            drawer.text((10, 10), text, fill="black", font=font)
            return image

        w1, w2, w3, w4 = 1000, 2000, 3000, 4000
        data_1, data_2 = self.f0403_上表面数据, self.f0404_下表面数据
        data_ua, data_ub, data_da, data_db = self.f0203_UA放缩, self.f0204_UB放缩, self.f0201_DA放缩, self.f0202_DB放缩
        elevation_range = 5
        im = Image.new('RGB', (w4, w4))

        im.paste(添加文本(depth_matrix_to_elevation_image(data_1, v_range=elevation_range), "上表面高程"), (0, 0))
        im.paste(添加文本(depth_matrix_to_rgb_image(data_1), "上表面RGB"), (w1, 0))
        im.paste(添加文本(depth_matrix_to_elevation_image(data_2, v_range=elevation_range), "下表面高程"), (0, w1))
        im.paste(添加文本(depth_matrix_to_rgb_image(data_2), "下表面RGB"), (w1, w1))

        im.paste(添加文本(depth_matrix_to_elevation_image(data_ua, v_range=elevation_range), "UA高程"), (w2, 0))
        im.paste(添加文本(depth_matrix_to_elevation_image(data_ub, v_range=elevation_range), "UB高程"), (w3, 0))
        im.paste(添加文本(depth_matrix_to_elevation_image(data_da, v_range=elevation_range), "DA高程"), (w2, w1))
        im.paste(添加文本(depth_matrix_to_elevation_image(data_db, v_range=elevation_range), "DB高程"), (w3, w1))

        im.paste(添加文本(depth_matrix_to_rgb_image(data_ua), "UA RGB"), (w2, w2))
        im.paste(添加文本(depth_matrix_to_rgb_image(data_ub), "UB RGB"), (w3, w2))
        im.paste(添加文本(depth_matrix_to_rgb_image(data_da), "DA RGB"), (w2, w3))
        im.paste(添加文本(depth_matrix_to_rgb_image(data_db), "DB RGB"), (w3, w3))

        delta = data_1 - data_2
        im.paste(添加文本(depth_matrix_to_elevation_image(delta, v_range=1), "Δh 1mm"), (0, w2))
        im.paste(添加文本(depth_matrix_to_elevation_image(delta, v_range=2), "Δh 2mm"), (w1, w2))
        im.paste(添加文本(depth_matrix_to_elevation_image(delta, v_range=5), "Δh 5mm"), (0, w3))
        im.paste(添加文本(depth_matrix_to_elevation_image(delta, v_range=10), "Δh 10mm"), (w1, w3))

        return im


if __name__ == '__main__':
    s25030102_劈裂面形貌扫描对比.main()
