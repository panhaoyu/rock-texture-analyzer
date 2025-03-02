import logging
from pathlib import Path

from PIL import Image

from batch_processor import BatchProcessor, mark_as_npy, mark_as_png
from batch_processor.processors.base import ManuallyProcessRequiredException
from rock_texture_analyzer.other_utils import depth_matrix_to_elevation_image, depth_matrix_to_rgb_image

logger = logging.getLogger(Path(__file__).name)

from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import zoom
import numpy as np


def remove_black_borders(array: np.ndarray) -> np.ndarray:
    while True:
        if (nan := np.isnan(array[:, 0, 0])).sum() / nan.size > 0.01:
            array = array[:, 1:, :]
        elif (nan := np.isnan(array[:, -1, 0])).sum() / nan.size > 0.01:
            array = array[:, :-1, :]
        elif (nan := np.isnan(array[0, :, 0])).sum() / nan.size > 0.01:
            array = array[1:, :, :]
        elif (nan := np.isnan(array[-1, :, 0])).sum() / nan.size > 0.01:
            array = array[:-1, :, :]
        else:
            break
    return array


def fill_nan_values(array: np.ndarray) -> np.ndarray:
    layers: list[np.ndarray] = []
    for i in range(array.shape[-1]):
        if np.any(np.isnan((layer := array[:, :, i].copy()))):
            mask = np.isnan(layer)
            interp = NearestNDInterpolator(np.argwhere(~mask), layer[~mask])
            layer[mask] = interp(np.argwhere(mask))
        layers.append(layer)
    return np.dstack(layers)


def scale_array(array: np.ndarray, target_size: tuple[int, int] = (1000, 1000)) -> np.ndarray:
    scale = (target_size[0] / array.shape[0], target_size[1] / array.shape[1])
    # noinspection PyTypeChecker
    return np.dstack([zoom(array[:, :, i], scale, order=2) for i in range(array.shape[-1])])


def process(array: np.ndarray) -> np.ndarray:
    array = remove_black_borders(array)
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
