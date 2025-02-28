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
from numpy import ndarray
from open3d.cpu.pybind.geometry import PointCloud

from p3_表面显微镜扫描数据处理.config import base_dir
from rock_texture_analyzer.point_cloud import write_point_cloud, draw_point_cloud, read_point_cloud


class ProcessMethod(typing.Callable):
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
        assert self.suffix is not None

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.func_name

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
        input_path = self.get_input_path(path)
        match input_path.suffix:
            case '.ply':
                return read_point_cloud(input_path)
            case '.pickle':
                with input_path.open('rb') as f:
                    return pickle.load(f)
            case '.npy' | '.npz':
                return np.load(input_path)
            case '.png' | '.jpg':
                with Image.open(input_path) as img:
                    return img.copy()
            case _:
                raise NotImplementedError(f"Unsupported file type: {input_path.suffix}")

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
        sorted_methods = sorted(class_methods.items(), key=lambda x: x[0])
        methods = [method for _, methods in sorted_methods for method in methods]
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
        for func in self.step_functions:
            output_path: Path = func.get_input_path(path)
            if output_path.exists():
                recreate_require = func.is_recreate_required
                if not recreate_require:
                    continue
            func_index, func_name = re.fullmatch(r'f(\d+)_(.*?)', func.func_name).groups()
            func_index = int(func_index)
            func.is_single_thread and self._single_thread_lock.acquire()
            try:
                result = func(self, output_path)
                func.write(result, output_path)
            except ManuallyProcessRequiredException as exception:
                message = exception.args or ()
                message = ''.join(message)
                self.print_safe(f'{func_index:02d} {path:10} {func_name} 需要人工处理：{message}')
                break
            except Exception as e:
                self.print_safe(f'{func_index:02d} {path:10} {func_name} 异常：{e}')
                with self._print_lock:
                    traceback.print_exc()
                break
            finally:
                func.is_single_thread and self._single_thread_lock.release_lock()
            self.print_safe(f'{func_index:02d} {path:10} {func_name} 完成')

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


class JpgProcessor(ProcessMethod):
    def read(self, path: Path) -> Image.Image:
        return super().read(path)

    def write(self, obj: Image.Image, path: Path) -> None:
        return super().write(obj, path)


def mark_as_jpg(func: Callable) -> JpgProcessor:
    func = JpgProcessor.of(func)
    assert func.suffix is None
    func.suffix = '.jpg'
    return func


class PngProcessor(ProcessMethod):
    def read(self, path: Path) -> Image.Image:
        return super().read(path)

    def write(self, obj: Image.Image, path: Path) -> None:
        return super().write(obj, path)


def mark_as_png(func: Callable) -> PngProcessor:
    func = PngProcessor.of(func)
    assert func.suffix is None
    func.suffix = '.png'
    return func


class PlyProcessor(ProcessMethod):
    def read(self, path: Path) -> PointCloud:
        return super().read(path)

    def write(self, obj: PointCloud, path: Path) -> None:
        return super().write(obj, path)


def mark_as_ply(func: Callable) -> PlyProcessor:
    func = PlyProcessor.of(func)
    assert func.suffix is None
    func.suffix = '.ply'
    return func


class NpyProcessor(ProcessMethod):
    def read(self, path: Path) -> ndarray:
        return super().read(path)

    def write(self, obj: ndarray, path: Path) -> None:
        return super().write(obj, path)


def mark_as_npy(func: Callable) -> NpyProcessor:
    func = NpyProcessor.of(func)
    assert func.suffix is None
    func.suffix = '.npy'
    return func


class PickleProcessor(ProcessMethod):
    def read(self, path: Path) -> typing.Any:
        return super().read(path)

    def write(self, obj: typing.Any, path: Path) -> None:
        return super().write(obj, path)


def mark_as_pickle(func: Callable) -> PickleProcessor:
    func = PickleProcessor.of(func)
    assert func.suffix is None
    func.suffix = '.pickle'
    return func
