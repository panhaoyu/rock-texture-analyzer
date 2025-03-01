from typing import Callable

from PIL import Image

from batch_processor.processors.base import BaseProcessMethod


class __JpgProcessor(BaseProcessMethod[Image.Image]):
    pass


def mark_as_jpg(func: Callable) -> __JpgProcessor:
    func = __JpgProcessor.of(func)
    assert func.suffix is None, func.suffix
    func.suffix = '.jpg'
    return func
