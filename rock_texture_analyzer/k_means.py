import numpy as np
from sklearn.cluster import KMeans


def get_centers_with_extension(
        data: np.ndarray,
        n_clusters: int = 2,
        extension_ratio: float = 0.1
) -> tuple[float, float]:
    """获取数据聚类中心点并扩展范围"""
    kmeans: KMeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data.reshape(-1, 1))
    centers = sorted(c for c in kmeans.cluster_centers_.flatten())
    min_val, max_val = min(centers), max(centers)
    extent = (max_val - min_val) * (1 + extension_ratio)
    return min_val - extent, max_val + extent
