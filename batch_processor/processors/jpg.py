from typing import Callable

from PIL import Image

from batch_processor.processors.base import BaseProcessor


class __JpgProcessor(BaseProcessor[Image.Image]):
    pass


def mark_as_jpg(func: Callable) -> __JpgProcessor:
    func = __JpgProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.jpg'
    return func
