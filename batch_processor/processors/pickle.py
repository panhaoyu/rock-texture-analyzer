import typing
from typing import Callable

from batch_processor.processors.base import BaseProcessMethod


class __PickleProcessor(BaseProcessMethod[typing.Any]):
    pass


def mark_as_pickle(func: Callable) -> __PickleProcessor:
    func = __PickleProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.pickle'
    return func
