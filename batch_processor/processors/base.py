import re
import threading
import typing
from functools import cached_property
from pathlib import Path
from typing import Callable

from batch_processor.batch_processor import BatchProcessor

T = typing.TypeVar('T')


class BaseProcessMethod(typing.Generic[T]):
    # 用于给各个装饰器使用的变量
    is_recreate_required: bool = False
    is_source: bool = False
    is_final: bool = False
    is_single_thread: bool = False
    suffix: str = None
    processor: 'BatchProcessor'
    single_thread_process_lock = threading.Lock()
    meta_process_lock = threading.Lock()

    # 用于记录目前已经处理以及未处理的文件列表
    all_stems: set[str]
    pending_stems: set[str]
    processed_stems: set[str]

    def __init__(self, func: Callable[[Path], typing.Any]):
        self.func = func
        self.func_name = func.__name__
        self.all_stems = set()
        self.pending_stems = set()
        self.processed_stems = set()

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

    def _read(self, path: Path):
        raise NotImplementedError

    def read(self, path: Path):
        path = self.get_input_path(path)
        self._read(path)

    def _write(self, obj: typing.Any, path: Path):
        raise NotImplementedError

    def write(self, obj: typing.Any, path: Path):
        path = self.get_input_path(path)
        self._write(obj, path)

    def on_batch_started(self):
        """
        在处理第一个文件之前，调用此回调。

        该回调的主要作用为：
        - 创建处理环境
        - 创建处理过程中需要的一些临时变量
        - 打开句柄
        """

    def on_batch_finished(self):
        """
        在处理最后一个文件之后，调用此回调

        该回调的主要作用为：
        - 清理处理环境
        - 清理处理过程中需要的一些临时变量
        - 关闭句柄
        """


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
