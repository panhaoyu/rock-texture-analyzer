import numpy as np
from scipy.ndimage import zoom


def scale_array(array: np.ndarray, target_size: tuple[int, int] = (1000, 1000)) -> np.ndarray:
    scale = (target_size[0] / array.shape[0], target_size[1] / array.shape[1])
    # noinspection PyTypeChecker
    return np.dstack([zoom(array[:, :, i], scale, order=2) for i in range(array.shape[-1])])
