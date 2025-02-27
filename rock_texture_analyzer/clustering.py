from typing import Union

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


class ValueDetectionError(ValueError):
    """当无法检测到预期的特征值时引发的异常"""
    pass


def find_two_peaks(data: np.ndarray, prominence: Union[float, list[float]]) -> tuple[float, float]:
    """去除背景数据后提取双峰分布的两个主要峰值，基于KDE方法。

    Args:
        data: 输入的原始数据
        prominence: 峰值检测时的最小显著性，支持单值或列表形式
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

    prominences = [prominence] if isinstance(prominence, (int, float)) else prominence
    for p in prominences:
        peaks, _ = find_peaks(density, prominence=p)
        if len(peaks) >= 2:
            peak_centers = x_range[peaks].flatten()
            return tuple(sorted(peak_centers[:2]))
    raise ValueDetectionError(
        f"无法检测到两个明显峰，尝试prominence列表{prominences}后仍失败")


def find_peaks_on_both_sides(
        point_x: np.ndarray,
        point_y: np.ndarray,
        thresholds: list[float],
) -> tuple[float, float, float, float]:
    """通过尝试不同阈值获取有效聚类中心"""
    x1, x2 = find_two_peaks(point_x, prominence=thresholds)
    y1, y2 = find_two_peaks(point_y, prominence=thresholds)
    return x1, x2, y1, y2


def process_clusters(
        data: np.ndarray,
        n_clusters: int = 2,
        *,
        extension_ratio: float | None = None
) -> tuple[float, ...] | tuple[float, float]:
    """处理聚类操作并返回扩展范围或聚类中心"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data.reshape(-1, 1))
    # noinspection PyUnresolvedReferences
    centers = sorted(c[0] for c in kmeans.cluster_centers_)

    return (
        (
            min(centers) - (max(centers) - min(centers)) * extension_ratio,
            max(centers) + (max(centers) - min(centers)) * extension_ratio
        ) if extension_ratio is not None else tuple(centers)
    )
