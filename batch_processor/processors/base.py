import pickle
import re
import threading
import typing
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from open3d.cpu.pybind.geometry import PointCloud

from batch_processor.batch_processor import BatchProcessor
from rock_texture_analyzer.point_cloud import read_point_cloud, write_point_cloud, draw_point_cloud

T = typing.TypeVar('T')


class BaseProcessMethod(typing.Generic[T]):
    # 用于给各个装饰器使用的变量
    is_recreate_required: bool = False
    is_source: bool = False
    is_final: bool = False
    is_single_thread: bool = False
    suffix: str = None
    processor: 'BatchProcessor'
    lock = threading.Lock()

    def __init__(self, func: Callable[[Path], typing.Any]):
        self.func = func
        self.func_name = func.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.func_name

    def __repr__(self):
        return f'<{self.func_name}[{self.suffix}]>'

    @cached_property
    def step_index(self):
        return int(re.fullmatch(f'f(\d+)_(.*)', self.func_name).group(1))

    @classmethod
    def of(cls, value: Callable | 'BaseProcessMethod'):
        if isinstance(value, BaseProcessMethod):
            return value
        else:
            return BaseProcessMethod(value)

    @cached_property
    def directory(self):
        return self.processor.base_dir / self.func_name.replace('_', '-').lstrip('f')

    def get_input_path(self, output_path: Path):
        return self.directory.joinpath(f'{output_path.stem}{self.suffix}')

    def read(self, path: Path):
        path = self.get_input_path(path)
        match self.suffix:
            case '.ply':
                return read_point_cloud(path)
            case '.pickle':
                with path.open('rb') as f:
                    return pickle.load(f)
            case '.npy' | '.npz':
                return np.load(path)
            case '.png' | '.jpg':
                with Image.open(path) as img:
                    return img.copy()
            case other:
                raise NotImplementedError(f"Unsupported file type: {other}")

    def write(self, obj: typing.Any, path: Path):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.get_input_path(path)
        match self.suffix:
            case '.ply':
                assert isinstance(obj, PointCloud)
                write_point_cloud(path, obj)
            case '.npy':
                assert isinstance(obj, np.ndarray)
                np.save(path, obj)
            case '.png':
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
            case '.xlsx':
                assert isinstance(obj, pd.DataFrame)
                obj.to_excel(path)
            case '.pickle':
                with path.open('wb') as f:
                    # noinspection PyTypeChecker
                    pickle.dump(obj, f)
            case other:
                raise NotImplementedError(f'Unknown suffix: "{other}"')

    def on_batch_start(self):
        raise NotImplementedError

    def on_batch_finished(self):
        raise NotImplementedError


class ManuallyProcessRequiredException(Exception):
    pass


def mark_as_recreate(func: Callable):
    func = BaseProcessMethod.of(func)
    func.is_recreate_required = True
    return func


def mark_as_source(func: Callable):
    func = BaseProcessMethod.of(func)
    func.is_source = True
    return func


def mark_as_final(func: Callable):
    func = BaseProcessMethod.of(func)
    func.is_final = True
    return func


def mark_as_single_thread(func: Callable):
    func = BaseProcessMethod.of(func)
    func.is_single_thread = True
    return func
