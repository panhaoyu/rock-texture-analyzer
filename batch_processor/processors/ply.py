import typing
from pathlib import Path
from typing import Callable

from open3d.cpu.pybind.geometry import PointCloud

from batch_processor.processors.base import BaseProcessMethod
from rock_texture_analyzer.point_cloud import read_point_cloud, write_point_cloud


class __PlyProcessor(BaseProcessMethod[PointCloud]):
    def _read(self, path: Path):
        return read_point_cloud(path)

    def _write(self, obj: typing.Any, path: Path):
        assert isinstance(obj, PointCloud)
        write_point_cloud(path, obj)


def mark_as_ply(func: Callable) -> __PlyProcessor:
    func = __PlyProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.ply'
    return func
