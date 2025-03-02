import logging
import re
import threading
import typing
from functools import cached_property
from pathlib import Path
from typing import Callable

from batch_processor.batch_processor import BatchManager

if typing.TYPE_CHECKING:
    from ..batch_processor import SerialProcess, ManuallyProcessRequiredException

T = typing.TypeVar('T')

logger = logging.getLogger(Path(__file__).stem)


class BaseProcessor(typing.Generic[T]):
    # 用于给各个装饰器使用的变量
    is_recreate_required: bool = False
    is_source: bool = False
    is_final: bool = False
    is_single_thread: bool = False
    suffix: str = None
    processor: 'BatchManager'

    def __init__(self, func: Callable[[], typing.Any]):
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

    def __str__(self):
        return self.func_name

    def __get__(self, instance: 'SerialProcess', owner) -> T:
        if not self.is_processed(instance.path):  # 总是先写入再读取，这样可以获取得到规范化后的数据类型
            obj = self.func()
            self.write(obj, instance.path)
        return self.read(instance.path)

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
        return self.processor.manager.base_dir / self.func_name.replace('_', '-').lstrip('f')

    def get_input_path(self, path: Path):
        return self.directory.joinpath(f'{path.stem}{self.suffix}')

    def is_processed(self, path: Path) -> bool:
        return self.get_input_path(path).exists()

    def _read(self, path: Path) -> T:
        raise NotImplementedError(self.__class__.__name__)

    def read(self, path: Path) -> T:
        path = self.get_input_path(path)
        return self._read(path)

    def _write(self, obj: T, path: Path):
        raise NotImplementedError(self.__class__.__name__)

    def write(self, obj: T, path: Path):
        path = self.get_input_path(path)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write(obj, path)

    def process(self, instance: 'SerialProcess') -> bool:
        # 返回值为函数的执行状态，如果正确执行，则返回True，如果报错，则返回False
        path = self.get_input_path(instance.path)
        stem = path.stem
        func_index, func_name = self.step_index, self.func_name
        self.is_single_thread and self.single_thread_process_lock.acquire_lock()
        self.pending_stems.remove(stem)
        self.processing_stems.add(stem)
        try:
            self.check_batch_started()
            if not self.is_processed(path) or self.is_recreate_required:
                getattr(instance, self.func_name)
        except ManuallyProcessRequiredException as exception:
            message = exception.args or ()
            message = ''.join(message)
            logger.info(f'{func_index:04d} {stem:10} {func_name} 需要人工处理：{message}')
            return False
        except Exception as e:
            logger.exception(f'{func_index:04d} {stem:10} {func_name}：{e}')
            return False
        finally:
            self.is_single_thread and self.single_thread_process_lock.release_lock()
        finished_stems = len(self.processed_stems) + 1
        all_stems = len(self.all_stems)
        logger.info(f'{func_index:04d} {stem:10} {finished_stems}/{all_stems} {func_name}')
        self.processing_stems.remove(stem)
        self.processed_stems.add(stem)
        self.check_batch_finished()
        return True

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
