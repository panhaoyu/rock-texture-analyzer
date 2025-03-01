import typing
from pathlib import Path
from typing import Callable

import numpy as np

from batch_processor.processors.base import BaseProcessMethod


class __NpyProcessor(BaseProcessMethod[np.ndarray]):
    def _read(self, path: Path):
        return np.load(path)

    def _write(self, obj: typing.Any, path: Path):
        assert isinstance(obj, np.ndarray)
        np.save(path, obj)


def mark_as_npy(func: Callable) -> __NpyProcessor:
    func = __NpyProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.npy'
    return func
