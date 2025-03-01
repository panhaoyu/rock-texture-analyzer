import typing
from functools import cached_property
from pathlib import Path
from typing import Callable

from batch_processor.processors.base import BaseProcessor


def mark_as_manual_combined_excel(columns: tuple[str, ...]):
    def wrapper(func: Callable):
        func = __CombinedExcelProcessor.of(func)
        assert func.suffix is None, func.suffix
        func.suffix = '.xlsx'
        func.columns = columns
        return func

    return wrapper


_SupportedTypes = typing.Union[str, float, int]
_RowType = tuple[_SupportedTypes, ...]


class __CombinedExcelProcessor(BaseProcessor[_RowType]):
    """全部的数据都被存储在一个excel里面，其中每一行代表一个试样"""
    columns: tuple[str, ...]
    suffix = '.xlsx'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # stem -> column values
        self.data: dict[str, _RowType] = {}

    def read(self, path: Path):
        return self.data[path.stem]

    def write(self, obj: typing.Any, path: Path):
        self.data[path.stem] = obj

    @cached_property
    def combined_file_path(self):
        return self.directory / f'combined{self.suffix}'

    def on_batch_started(self):
        # todo 从 combined_file 里面读取数据，然后写入到data里面去
        raise NotImplementedError

    def on_batch_ended(self):
        # todo 将data里面的数据写入到 combined_file
        raise NotImplementedError
