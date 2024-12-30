import abc
import functools
import pickle
import re
import types
from abc import ABC
from collections import UserDict
from functools import cached_property
from pathlib import Path
from typing import Callable, Any

import numpy as np
import open3d.cpu.pybind.io
from open3d.cpu.pybind.geometry import PointCloud
from pypinyin.core import lazy_pinyin


class InnerCache(UserDict):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def get_new_file_path(self, name: str, value: Any) -> Path:
        suffix = '.pickle'
        if isinstance(value, np.ndarray):
            suffix = '.npy'
        elif isinstance(value, PointCloud):
            suffix = '.ply'
        return self.path.joinpath(f'{name}{suffix}')

    def get_existing_file_path(self, name: str) -> Path:
        files = list(self.path.glob(f'{name}.*'))
        assert len(files) < 2
        return files[0] if files else None

    def get_cache_file_path(self, name: str, value: Any = None):
        assert isinstance(name, str), type(name)
        if value is None:
            result = self.get_existing_file_path(name)
        else:
            result = self.get_new_file_path(name, value)
        name = result.stem
        name = '_'.join(lazy_pinyin(name))
        name = re.sub(r'_+', '_', name)
        result = result.with_stem(name)
        return result

    def __contains__(self, item):
        files = list(self.path.glob(f'{item}.*'))
        assert len(files) < 2
        return bool(files)

    def __getitem__(self, item):
        path = self.get_cache_file_path(item)
        match path.suffix:
            case '.pickle':
                with open(path, 'rb') as f:
                    return pickle.load(f)
            case '.npy':
                return np.load(path)
            case '.ply':
                return open3d.io.read_point_cloud(path.as_posix())
            case _:
                raise ValueError(f'Unknown suffix of key "{item}"')

    def __setitem__(self, key, value):
        path = self.get_cache_file_path(key, value)
        match path.suffix:
            case '.pickle':
                with open(path, 'wb') as f:
                    pickle.dump(value, f)
            case '.npy':
                np.save(path, value)
            case '.ply':
                open3d.cpu.pybind.io.write_point_cloud(path.as_posix(), value)
            case _:
                raise ValueError(f'Unknown suffix of key "{key}"')

    def __delitem__(self, key):
        self.get_cache_file_path(key).unlink(missing_ok=True)

    def __iter__(self):
        files = [
            *self.path.glob('*.pickle'),
            *self.path.glob('*.npy'),
            *self.path.glob('*.ply'),
        ]
        return {i.stem for i in files}


class MethodCache(ABC):
    @abc.abstractmethod
    def get_cache_folder(self) -> Path:
        raise NotImplementedError

    @cached_property
    def cache_dict(self) -> dict | UserDict:
        raise NotImplementedError


class MethodDiskCache(MethodCache, ABC):
    @cached_property
    def cache_dict(self) -> dict | UserDict:
        return InnerCache(self.get_cache_folder())


def method_cache(func: Callable):
    key = func.__name__
    assert isinstance(func, (types.FunctionType,)), type(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert len(args) == 1 and not kwargs, 'Only methods without parameters are supported'

        self: MethodCache = args[0]
        assert isinstance(self, MethodCache), f'Only subclass of MethodCache supported'

        if key in self.cache_dict:
            return self.cache_dict[key]

        result = func(self)
        self.cache_dict[key] = result

        return result

    return wrapper
