import numpy as np
from numpy.typing import NDArray


def _tilt_correction_inner(arr1: NDArray, arr2: NDArray) -> tuple[NDArray, NDArray]:
    arr1, arr2 = arr1.copy(), arr2.copy()
    layer1, layer2 = arr1[:, :, 0], arr2[:, :, 0]
    rows, cols = layer1.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    A = np.column_stack((X.ravel(), Y.ravel(), np.ones(X.size)))
    b = (layer1 - layer2).ravel()

    # 初始拟合计算残差并排除匹配不好的数据点
    coefficients = np.linalg.lstsq(A, b, rcond=None)[0]
    residuals = np.abs(b - A @ coefficients)
    mask = residuals < np.percentile(residuals, 90)

    # 选取有效点再进行真正的最小二乘拟合
    coefficients = np.linalg.lstsq(A[mask], b[mask], rcond=None)[0]
    reverse = (A @ coefficients).reshape(layer1.shape) / 2
    layer1 -= reverse
    layer2 += reverse
    arr1[:, :, 0], arr2[:, :, 0] = layer1, layer2
    return arr1, arr2


def tilt_correction(arr1: NDArray, arr2: NDArray) -> tuple[NDArray, NDArray]:
    for _ in range(5):
        arr1, arr2 = _tilt_correction_inner(arr1, arr2)
    return arr1, arr2
