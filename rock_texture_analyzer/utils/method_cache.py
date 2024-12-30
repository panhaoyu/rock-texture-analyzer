import abc
import functools
import pickle
import types
from abc import ABC
from collections import UserDict
from functools import cached_property
from pathlib import Path
from typing import Callable


class InnerCache(UserDict):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def get_cache_file_path(self, name: str):
        assert isinstance(name, str), type(name)
        return self.path / f'{name}.pickle'

    def __contains__(self, item):
        return self.get_cache_file_path(item).exists()

    def __getitem__(self, item):
        path = self.get_cache_file_path(item)
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result

    def __setitem__(self, key, value):
        path = self.get_cache_file_path(key)
        with open(path, 'wb') as f:
            pickle.dump(value, f)
        self.data[key] = value

    def __delitem__(self, key):
        self.get_cache_file_path(key).unlink(missing_ok=True)

    def __iter__(self):
        return [i.stem for i in self.path.glob('*.pickle')]


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
