import numpy as np
from PIL import Image


def boolean_to_image(array: np.ndarray) -> Image.Image:
    array = np.array([[0, 0, 0], [255, 255, 255]])[array]
    array = array.astype(np.uint8)
    return Image.fromarray(array)
