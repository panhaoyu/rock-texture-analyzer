import logging
import typing
from functools import cached_property
from pathlib import Path
from typing import Callable

import pandas as pd

from batch_processor.processors.base import BaseProcessor

logger = logging.getLogger(Path(__file__).stem)

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

    def _read(self, path: Path):
        return self.data.get(path.stem, None)

    def _write(self, obj: typing.Any, path: Path):
        self.data[path.stem] = obj

    def is_processed(self, path: Path) -> bool:
        return self.read(path) is not None

    @cached_property
    def combined_file_path(self):
        return self.directory / f'combined{self.suffix}'

    def on_batch_started(self):
        if not self.combined_file_path.exists():
            return

        df = pd.read_excel(self.combined_file_path, engine='openpyxl')
        df_columns = set(df.columns.tolist()) - {'stem'}
        if set(df_columns) != set(self.columns):
            raise ValueError(f"Column mismatch. Expected {self.columns}, got {df_columns}")

        for _, row in df.iterrows():
            stem = row['stem']
            self.data[stem] = tuple(row[col] for col in self.columns)

    def on_batch_finished(self):
        if not self.data:
            return
        items = sorted(self.data.items(), key=lambda x: x[0])
        df = pd.DataFrame.from_dict(
            {k: list(v) for k, v in items},
            orient='index',
            columns=self.columns
        )
        df.index.name = 'stem'
        df.to_excel(self.combined_file_path, index=True)


_CT = typing.TypeVar('_CT')


def mark_as_combined_excel(columns: tuple[str, ...]) -> Callable[[_CT], __CombinedExcelProcessor]:
    def wrapper(func: _CT) -> __CombinedExcelProcessor:
        func = __CombinedExcelProcessor.of(func)
        func.columns = columns
        return func

    return wrapper
