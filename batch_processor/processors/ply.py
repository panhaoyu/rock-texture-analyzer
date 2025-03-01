import typing
from pathlib import Path
from typing import Callable

from open3d.cpu.pybind.geometry import PointCloud

from batch_processor.processors.base import BaseProcessor
from rock_texture_analyzer.point_cloud import read_point_cloud, write_point_cloud


class __PlyProcessor(BaseProcessor[PointCloud]):
    suffix = '.ply'

    def _read(self, path: Path):
        return read_point_cloud(path)

    def _write(self, obj: typing.Any, path: Path):
        assert isinstance(obj, PointCloud)
        write_point_cloud(path, obj)


def mark_as_ply(func: Callable) -> __PlyProcessor:
    return __PlyProcessor.of(func)
