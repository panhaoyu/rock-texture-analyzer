import pickle
import typing
from pathlib import Path
from typing import Callable

from batch_processor.processors.base import BaseProcessMethod


class __PickleProcessor(BaseProcessMethod[typing.Any]):
    def _read(self, path: Path):
        with path.open('rb') as f:
            return pickle.load(f)

    def _write(self, obj: typing.Any, path: Path):
        with path.open('wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(obj, f)


def mark_as_pickle(func: Callable) -> __PickleProcessor:
    func = __PickleProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.pickle'
    return func
