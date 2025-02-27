import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity


class ValueDetectionError(ValueError):
    """当无法检测到预期的特征值时引发的异常"""
    pass

def get_two_main_value_filtered(data: np.ndarray, prominence: float = 0.05) -> tuple[float, float]:
    """去除背景数据后提取双峰分布的两个主要峰值，基于KDE方法。

    Args:
        data: 输入的原始数据
        prominence: 峰值检测时的最小显著性

    Returns:
        两个主峰对应的位置

    Raises:
        ValueDetectionError: 当无法检测到两个有效峰时
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
    kde.fit(data.reshape(-1, 1))

    x_min, x_max = data.min(), data.max()
    x_range = x_max - x_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    x_range = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    log_density = kde.score_samples(x_range)
    density = np.exp(log_density)
    peaks = find_peaks(density, prominence=prominence)[0]

    if len(peaks) < 2:
        # plt.figure()
        # plt.hist(data, bins=100)
        # plt.show()
        # plt.figure()
        # plt.plot(density)
        # plt.show()
        raise ValueDetectionError(f"无法检测到两个明显的峰，找到的峰数量：{len(peaks)}")

    peak_centers = x_range[peaks - 1].flatten()
    # noinspection PyTypeChecker
    return tuple(sorted(peak_centers.tolist()))
