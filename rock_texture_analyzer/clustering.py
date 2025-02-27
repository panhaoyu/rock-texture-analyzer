import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


def find_valid_clusters(
        point_x: np.ndarray,
        point_y: np.ndarray,
        thresholds: list[float]
) -> tuple[float, float, float, float]:
    """通过尝试不同阈值获取有效聚类中心"""
    for threshold in thresholds:
        try:
            x1, x2 = get_two_main_value_filtered(point_x, threshold)
            y1, y2 = get_two_main_value_filtered(point_y, threshold)
            return x1, x2, y1, y2
        except ValueDetectionError:
            continue
    raise ValueDetectionError(f"无法找到有效阈值，尝试了所有阈值: {thresholds}")


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
    peaks: np.ndarray = find_peaks(density, prominence=prominence)[0]

    if len(peaks) < 2:
        raise ValueDetectionError(f"无法检测到两个明显的峰，找到的峰数量：{len(peaks)}")

    peak_centers = x_range[peaks - 1].flatten()
    # noinspection PyTypeChecker
    return tuple(sorted(peak_centers.tolist()))
