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
from rock_texture_analyzer.point_cloud import read_point_cloud, write_point_cloud, draw_point_cloud


class ProcessMethod(typing.Callable):
    # 用于给各个装饰器使用的变量
    is_recreate_required: bool = False
    is_source: bool = False
    is_final: bool = False
    is_single_thread: bool = False
    suffix: str = ".png"

    def __init__(self, func: Callable[[Path], typing.Any]):
        self.func = func
        self.func_name = func.__name__

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
        return [method for _, methods in sorted_methods for method in methods]

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

    def process_stem(self, stem: str) -> None:
        for func in self.step_functions:
            output_path: Path = self.get_file_path(func, stem)
            if output_path.exists():
                recreate_require = func.is_recreate_required
                if not recreate_require:
                    continue
            func_index, func_name = re.fullmatch(r'f(\d+)_(.*?)', func.func_name).groups()
            func_index = int(func_index)
            func.is_single_thread and self._single_thread_lock.acquire()
            try:
                result = func(self, output_path)
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                match output_path.suffix:
                    case '.ply':
                        assert isinstance(result, PointCloud)
                        write_point_cloud(output_path, result)
                    case '.npy':
                        assert isinstance(result, np.ndarray)
                        np.save(output_path, result)
                    case '.png':
                        if isinstance(result, np.ndarray):
                            result = Image.fromarray(result)
                        if isinstance(result, plt.Figure):
                            result.savefig(output_path)
                        elif isinstance(result, Image.Image):
                            result.save(output_path)
                        elif isinstance(result, PointCloud):
                            draw_point_cloud(result, output_path)
                        else:
                            raise NotImplementedError(f'Unknown png type: {type(result)}')
                    case '.xlsx':
                        assert isinstance(result, pd.DataFrame)
                        result.to_excel(output_path)
                    case '.pickle':
                        output_path.write_bytes(pickle.dumps(result))
                    case other:
                        raise NotImplementedError(f'Unknown suffix: "{other}"')
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

    def get_file_path(self, func: ProcessMethod, stem: str | Path) -> Path:
        if isinstance(stem, Path):
            stem = stem.stem
        dir_path: Path = self.base_dir / func.func_name.replace('_', '-').lstrip('f')
        extensions: set[str] = {p.suffix for p in dir_path.glob('*') if p.is_file()}
        suffix: str = next(iter(extensions), func.suffix)
        return dir_path / f'{stem}{suffix}'

    def get_input_path(self, func: ProcessMethod, output_path: Path) -> Path:
        return self.get_file_path(func, output_path.stem)

    def get_input_ply(self, func: ProcessMethod, output_path: Path):
        return read_point_cloud(self.get_input_path(func, output_path))

    def get_input_image(self, func: ProcessMethod, output_path: Path):
        input_path = self.get_input_path(func, output_path)
        with Image.open(input_path) as img:
            return img.copy()

    def get_input_array(self, func: ProcessMethod, output_path: Path):
        path = self.get_input_path(func, output_path)
        match path.suffix:
            case '.png' | '.jpg':
                image = self.get_input_image(func, output_path)
                array = np.asarray(image).copy()
            case '.npy':
                array = np.load(path)
            case other:
                raise NotImplementedError(other)
        return array

    enable_multithread: bool = True
    is_debug: bool = False

    @classmethod
    def main(cls) -> None:
        obj = cls()

        functions = obj.step_functions
        source_function = only((f for f in functions if f.is_source), functions[0])
        final_function = only((f for f in functions if f.is_final), functions[-1])

        for func in functions:
            obj.get_file_path(func, 'dummy').parent.mkdir(parents=True, exist_ok=True)

        source_file = obj.get_file_path(source_function, 'dummy')
        source_dir: Path = source_file.parent
        stems = [file.stem for file in source_dir.glob(f'*{source_file.suffix}')]
        if cls.is_debug:
            stems = stems[:2]
        if cls.enable_multithread:
            with ThreadPoolExecutor() as executor:
                executor.map(obj.process_stem, stems)
        else:
            for stem in stems:
                obj.process_stem(stem)
        zip_path = source_dir.parent / f"{source_dir.parent.name}.zip"
        final_dir = obj.get_file_path(final_function, 'dummy').parent
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


def mark_as_method(func: Callable) -> ProcessMethod:
    func = ProcessMethod.of(func)
    return func


def mark_as_final(func: Callable):
    func = ProcessMethod.of(func)
    func.is_final = True
    return func


def mark_as_single_thread(func: Callable):
    func = ProcessMethod.of(func)
    func.is_single_thread = True
    return func


def mark_as_ply(func: Callable):
    func = ProcessMethod.of(func)
    func.suffix = '.ply'
    return func


def mark_as_npy(func: Callable):
    func = ProcessMethod.of(func)
    func.suffix = '.npy'
    return func
