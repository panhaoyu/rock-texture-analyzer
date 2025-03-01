from typing import Callable

import numpy as np

from batch_processor.processors.base import BaseProcessMethod


class __NpyProcessor(BaseProcessMethod[np.ndarray]):
    pass


def mark_as_npy(func: Callable) -> __NpyProcessor:
    func = __NpyProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.npy'
    return func
