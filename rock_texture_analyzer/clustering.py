import numpy as np
from sklearn.cluster import KMeans

from rock_texture_analyzer.utils.get_two_peaks import (
    get_two_main_value_filtered,
    ValueDetectionError
)


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
