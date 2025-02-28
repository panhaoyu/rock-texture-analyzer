import pickle
import re
import threading
import traceback
import typing
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from more_itertools import only
from open3d.cpu.pybind.geometry import PointCloud

from p3_表面显微镜扫描数据处理.config import base_dir
from rock_texture_analyzer.point_cloud import write_point_cloud, draw_point_cloud, read_point_cloud

T = typing.TypeVar('T')


class ProcessMethod(typing.Callable, typing.Generic[T]):
    # 用于给各个装饰器使用的变量
    is_recreate_required: bool = False
    is_source: bool = False
    is_final: bool = False
    is_single_thread: bool = False
    suffix: str = None
    processor: 'BaseProcessor'

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
    def of(cls, value: Callable | 'ProcessMethod'):
        if isinstance(value, ProcessMethod):
            return value
        else:
            return ProcessMethod(value)

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
                    pickle.dump(obj, f)
            case other:
                raise NotImplementedError(f'Unknown suffix: "{other}"')


class BaseProcessor:
    _print_lock: threading.Lock = threading.Lock()
    _single_thread_lock: threading.Lock = threading.Lock()

    @classmethod
    def print_safe(cls, message: str) -> None:
        with cls._print_lock:
            print(message)

    @cached_property
    def step_functions(self):
        class_methods: dict[int, list[ProcessMethod]] = {}
        for klass in self.__class__.mro():
            for value in vars(klass).values():
                if isinstance(value, ProcessMethod):
                    class_methods.setdefault(value.step_index, []).append(value)
        # 按照书写的顺序进行排序，而不是按照名称进行排序
        methods = list(class_methods.values())
        for method in methods:
            method.processor = self
        return methods

    @cached_property
    def base_dir(self) -> Path:
        class_name = self.__class__.__name__
        match = re.fullmatch(r's(\d{8})_(.+)', class_name)
        if not match:
            raise ValueError(f"无效的类名: {class_name}")
        code, name = match.groups()
        result = base_dir.joinpath(f"{code}-{name}")
        if not result.exists():
            name = name.replace('_', '-')
            result = base_dir.joinpath(f"{code}-{name}")
        assert result.exists(), result
        return result

    def process_path(self, path: Path) -> None:
        stem = path.stem
        for func in self.step_functions:
            output_path: Path = func.get_input_path(path)
            if output_path.exists():
                recreate_require = func.is_recreate_required
                if not recreate_require:
                    continue
            func_index, func_name = func.step_index, func.func_name
            func.is_single_thread and self._single_thread_lock.acquire()
            try:
                result = func(self, output_path)
                func.write(result, output_path)
            except ManuallyProcessRequiredException as exception:
                message = exception.args or ()
                message = ''.join(message)
                self.print_safe(f'{func_index:02d} {stem:10} {func_name} 需要人工处理：{message}')
                break
            except Exception as e:
                self.print_safe(f'{func_index:02d} {stem:10} {func_name} 异常：{e}')
                with self._print_lock:
                    traceback.print_exc()
                break
            finally:
                func.is_single_thread and self._single_thread_lock.release_lock()
            self.print_safe(f'{func_index:02d} {stem:10} {func_name} 完成')

    enable_multithread: bool = True
    is_debug: bool = False

    @classmethod
    def main(cls) -> None:
        obj = cls()

        functions = obj.step_functions
        source_function = only((f for f in functions if f.is_source), functions[0])
        final_function = only((f for f in functions if f.is_final), functions[-1])

        files = [file for file in source_function.directory.glob(f'*{source_function.suffix}')]
        if cls.is_debug:
            files = files[:2]
        if cls.enable_multithread:
            with ThreadPoolExecutor() as executor:
                executor.map(obj.process_path, files)
        else:
            for file in files:
                obj.process_path(file)
        zip_path = obj.base_dir / f"{obj.base_dir.name}.zip"
        final_dir = final_function.directory
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            [zip_file.write(file, file.name) for file in final_dir.glob('*.png')]


class ManuallyProcessRequiredException(Exception):
    pass


def mark_as_recreate(func: Callable):
    func = ProcessMethod.of(func)
    func.is_recreate_required = True
    return func


def mark_as_source(func: Callable):
    func = ProcessMethod.of(func)
    func.is_source = True
    return func


def mark_as_final(func: Callable):
    func = ProcessMethod.of(func)
    func.is_final = True
    return func


def mark_as_single_thread(func: Callable):
    func = ProcessMethod.of(func)
    func.is_single_thread = True
    return func


class __JpgProcessor(ProcessMethod[Image.Image]): pass


class __PngProcessor(ProcessMethod[Image.Image]): pass


class __PlyProcessor(ProcessMethod[PointCloud]): pass


class __NpyProcessor(ProcessMethod[np.ndarray]): pass


class __PickleProcessor(ProcessMethod[typing.Any]): pass


def mark_as_jpg(func: Callable) -> __JpgProcessor:
    func = __JpgProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.jpg'
    return func


def mark_as_png(func: Callable) -> __PngProcessor:
    func = __PngProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.png'
    return func


def mark_as_ply(func: Callable) -> __PlyProcessor:
    func = __PlyProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.ply'
    return func


def mark_as_npy(func: Callable) -> __NpyProcessor:
    func = __NpyProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.npy'
    return func


def mark_as_pickle(func: Callable) -> __PickleProcessor:
    func = __PickleProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.pickle'
    return func
