import numpy as np
from numpy.typing import NDArray


def _tilt_correction_inner(arr1: NDArray, arr2: NDArray) -> tuple[NDArray, NDArray]:
    arr1, arr2 = arr1.copy(), arr2.copy()
    layer1, layer2 = arr1[:, :, 0], arr2[:, :, 0]
    diff = layer1 - layer2
    rows, cols = layer1.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    A = np.column_stack((X.ravel(), Y.ravel(), np.ones(X.size)))
    b = diff.ravel()
    # 初始拟合计算残差并筛选最佳50%数据点
    coeffs_initial = np.linalg.lstsq(A, b, rcond=None)[0]
    residuals = np.abs(b - A @ coeffs_initial)
    mask = residuals <= np.percentile(residuals, 50)
    coeffs = np.linalg.lstsq(A[mask], b[mask], rcond=None)[0]
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
