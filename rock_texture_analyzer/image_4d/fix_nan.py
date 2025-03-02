import numpy as np
from scipy.interpolate import NearestNDInterpolator


def remove_nan_borders(array: np.ndarray) -> np.ndarray:
    while True:
        if (nan := np.isnan(array[:, 0, 0])).sum() / nan.size > 0.01:
            array = array[:, 1:, :]
        elif (nan := np.isnan(array[:, -1, 0])).sum() / nan.size > 0.01:
            array = array[:, :-1, :]
        elif (nan := np.isnan(array[0, :, 0])).sum() / nan.size > 0.01:
            array = array[1:, :, :]
        elif (nan := np.isnan(array[-1, :, 0])).sum() / nan.size > 0.01:
            array = array[:-1, :, :]
        else:
            break
    return array


def fill_nan_values(array: np.ndarray) -> np.ndarray:
    layers: list[np.ndarray] = []
    for i in range(array.shape[-1]):
        if np.any(np.isnan((layer := array[:, :, i].copy()))):
            mask = np.isnan(layer)
            interp = NearestNDInterpolator(np.argwhere(~mask), layer[~mask])
            layer[mask] = interp(np.argwhere(mask))
        layers.append(layer)
    return np.dstack(layers)
