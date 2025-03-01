import typing
from pathlib import Path
from typing import Callable

import numpy as np

from batch_processor.processors.base import BaseProcessor


class __NpyProcessor(BaseProcessor[np.ndarray]):
    suffix = '.npy'

    def _read(self, path: Path):
        return np.load(path)

    def _write(self, obj: typing.Any, path: Path):
        assert isinstance(obj, np.ndarray)
        np.save(path, obj)


def mark_as_npy(func: Callable) -> __NpyProcessor:
    return __NpyProcessor.of(func)
