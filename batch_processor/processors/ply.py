from typing import Callable

from open3d.cpu.pybind.geometry import PointCloud

from batch_processor.processors.base import BaseProcessMethod


class __PlyProcessor(BaseProcessMethod[PointCloud]):
    pass


def mark_as_ply(func: Callable) -> __PlyProcessor:
    func = __PlyProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.ply'
    return func
