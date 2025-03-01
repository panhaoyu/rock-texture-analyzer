import typing
from typing import Callable

from batch_processor.processors.base import BaseProcessor


def mark_as_manual_combined_excel(columns: tuple[str, ...]):
    def wrapper(func: Callable):
        func = __ManualCombinedExcelProcessor.of(func)
        assert func.suffix is None, func.suffix
        func.suffix = '.xlsx'
        func.columns = columns
        return func

    return wrapper


class __ManualCombinedExcelProcessor(BaseProcessor[tuple[typing.Union[str, float, int], ...]]):
    """全部的数据都被存储在一个excel里面，其中每一行代表一个试样"""
    columns: tuple[str, ...]
