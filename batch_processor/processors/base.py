import re
import threading
import typing
from functools import cached_property
from pathlib import Path
from typing import Callable

if typing.TYPE_CHECKING:
    from ..batch_processor import BatchProcessor

T = typing.TypeVar('T')


class BaseProcessor(typing.Generic[T]):
    # 用于给各个装饰器使用的变量
    is_recreate_required: bool = False
    is_source: bool = False
    is_final: bool = False
    is_single_thread: bool = False
    suffix: str = None
    processor: 'BatchProcessor'

    def __init__(self, func: Callable[[Path], typing.Any]):
        self.func = func
        self.func_name = func.__name__

        # 处理处理单线程的事务
        self.single_thread_process_lock = threading.Lock()

        # 用于记录目前已经处理以及未处理的文件列表
        self.all_stems: set[str] = set()
        self.pending_stems: set[str] = set()
        self.processing_stems: set[str] = set()
        self.processed_stems: set[str] = set()

        # 用于记录相应的回调是否已经被调用
        self.meta_process_lock = threading.Lock()
        self.is_batch_started_called: bool = False
        self.is_batch_finished_called: bool = False

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
    def of(cls, value: Callable | 'BaseProcessor'):
        if isinstance(value, BaseProcessor):
            return value
        else:
            return cls(value)

    @cached_property
    def directory(self):
        return self.processor.base_dir / self.func_name.replace('_', '-').lstrip('f')

    def get_input_path(self, output_path: Path):
        return self.directory.joinpath(f'{output_path.stem}{self.suffix}')

    def is_processed(self, path: Path) -> bool:
        return self.get_input_path(path).exists()

    def _read(self, path: Path):
        raise NotImplementedError(self.__class__.__name__)

    def read(self, path: Path):
        path = self.get_input_path(path)
        return self._read(path)

    def _write(self, obj: typing.Any, path: Path):
        raise NotImplementedError(self.__class__.__name__)

    def write(self, obj: typing.Any, path: Path):
        path = self.get_input_path(path)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write(obj, path)

    def on_batch_started(self):
        """
        在处理第一个文件之前，调用此回调。

        该回调的主要作用为：
        - 创建处理环境
        - 创建处理过程中需要的一些临时变量
        - 打开句柄
        """

    def check_batch_started(self):
        with self.meta_process_lock:
            if self.is_batch_started_called:
                return
            self.is_batch_started_called = True
            self.on_batch_started()

    def on_batch_finished(self):
        """
        在处理最后一个文件之后，调用此回调

        该回调的主要作用为：
        - 清理处理环境
        - 清理处理过程中需要的一些临时变量
        - 关闭句柄
        """

    def check_batch_finished(self):
        with self.meta_process_lock:
            if self.is_batch_finished_called:
                return
            if self.processing_stems or self.pending_stems:
                return
            self.is_batch_finished_called = True
            self.on_batch_finished()


class ManuallyProcessRequiredException(Exception):
    pass


def mark_as_recreate(func: BaseProcessor):
    func.is_recreate_required = True
    return func


def mark_as_source(func: BaseProcessor):
    func.is_source = True
    return func


def mark_as_final(func: BaseProcessor):
    func.is_final = True
    return func


def mark_as_single_thread(func: BaseProcessor):
    func.is_single_thread = True
    return func
