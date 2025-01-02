import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity


def get_two_main_value_filtered(data: np.ndarray) -> tuple[float, float]:
    """
    去除背景数据后提取双峰分布的两个主要峰值，基于KDE方法.

    Args:
        data (np.ndarray): 输入数据.

    Returns:
        tuple[float, float]: 两个主要峰值的中心位置.
    """
    # 使用KDE估计数据的密度
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
    kde.fit(data.reshape(-1, 1))

    # 在数据范围内生成密度估计
    x_min, x_max = np.min(data), np.max(data)
    x_range = x_max - x_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    x_range = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    log_density = kde.score_samples(x_range)
    density = np.exp(log_density)

    # 通过查找密度的峰值来确定主要峰
    peaks = find_peaks(density, prominence=0.05)[0]

    if len(peaks) < 2:
        peaks = find_peaks(density, prominence=0.02)[0]

    if len(peaks) < 2:
        plt.figure()
        plt.hist(data, bins=100)
        plt.show()
        plt.figure()
        plt.plot(density)
        plt.show()
        raise ValueError("无法检测到两个明显的峰值，请调整参数.")

    # 返回峰值对应的 x 值作为两个主要峰的中心位置
    peak_centers = x_range[peaks - 1].flatten()

    # 返回按大小排序的两个峰值
    return tuple(sorted(peak_centers))
