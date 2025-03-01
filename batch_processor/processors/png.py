from typing import Callable

from PIL import Image

from batch_processor.processors.base import BaseProcessMethod


class __PngProcessor(BaseProcessMethod[Image.Image]):
    pass


def mark_as_png(func: Callable) -> __PngProcessor:
    func = __PngProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.png'
    return func
