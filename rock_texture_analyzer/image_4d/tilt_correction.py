import numpy as np
from numpy.typing import NDArray


def _tilt_correction_inner(arr1: NDArray, arr2: NDArray) -> tuple[NDArray, NDArray]:
    arr1, arr2 = arr1.copy(), arr2.copy()
    layer1, layer2 = arr1[:, :, 0], arr2[:, :, 0]
    diff = layer1 - layer2
    rows, cols = layer1.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    A = np.column_stack((X.ravel(), Y.ravel(), np.ones(X.size)))
    coeffs, residuals, _, _ = np.linalg.lstsq(A, diff.ravel(), rcond=None)
    plane = (coeffs[0] * X + coeffs[1] * Y + coeffs[2]).reshape(layer1.shape)
    reverse = plane / 2
    layer1 -= reverse
    layer2 += reverse
    arr1[:, :, 0], arr2[:, :, 0] = layer1, layer2
    return arr1, arr2


def tilt_correction(arr1: NDArray, arr2: NDArray) -> tuple[NDArray, NDArray]:
    for _ in range(5):
        arr1, arr2 = _tilt_correction_inner(arr1, arr2)
    return arr1, arr2
