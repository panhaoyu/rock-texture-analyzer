import typing
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from open3d.cpu.pybind.geometry import PointCloud

from batch_processor.processors.base import BaseProcessor
from rock_texture_analyzer.point_clode.point_cloud import draw_point_cloud


class __JpgProcessor(BaseProcessor[Image.Image]):
    suffix = '.jpg'

    def _read(self, path: Path):
        return Image.open(path)

    def _write(self, obj: typing.Any, path: Path):
        if isinstance(obj, np.ndarray):
            obj = Image.fromarray(obj)
        if isinstance(obj, plt.Figure):
            obj.savefig(path)
            plt.close(obj)
        elif isinstance(obj, Image.Image):
            obj.save(path)
        elif isinstance(obj, PointCloud):
            draw_point_cloud(obj, path)
        else:
            raise NotImplementedError(f'Unknown png type: {type(obj)}')


def mark_as_jpg(func: Callable) -> __JpgProcessor:
    return __JpgProcessor.of(func)
