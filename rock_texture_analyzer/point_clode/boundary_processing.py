import numpy as np

from rock_texture_analyzer.point_clode.clustering import find_two_peaks, ValueDetectionError, find_single_peak


def get_boundaries(points: np.ndarray) -> tuple[float, float, float, float, float, float]:
    _, _, z = points.T
    thresholds = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    try:
        z0, z1 = find_two_peaks(z, thresholds)
    except ValueDetectionError:
        z0, z1 = np.min(z), find_single_peak(z, thresholds)
    z_range = z1 - z0
    points = points[(z > (z0 + z_range * 0.1)) & (z < (z1 - z_range * 0.4))]
    x, y, _ = points.T

    x0c, x1c = find_two_peaks(x, thresholds)
    y0c, y1c = find_two_peaks(y, thresholds)
    assert y1c > y0c and x1c > x0c

    dx = 0.1 * (x1c - x0c)
    dy = 0.1 * (y1c - y0c)

    x0_pts = x[(np.abs(x - x0c) < dx) & (y > y0c + dy) & (y < y1c - dy)]
    x1_pts = x[(np.abs(x - x1c) < dx) & (y > y0c + dy) & (y < y1c - dy)]
    y0_pts = y[(np.abs(y - y0c) < dy) & (x > x0c + dx) & (x < x1c - dx)]
    y1_pts = y[(np.abs(y - y1c) < dy) & (x > x0c + dx) & (x < x1c - dx)]

    x0 = np.mean(x0_pts) + 5 * np.std(x0_pts)
    x1 = np.mean(x1_pts) - 5 * np.std(x1_pts)
    y0 = np.mean(y0_pts) + 5 * np.std(y0_pts)
    y1 = np.mean(y1_pts) - 5 * np.std(y1_pts)

    return x0, x1, y0, y1, z0, z1



def generate_axis_boundary_mask(
        data: np.ndarray,
        axis: int,
        center: float,
        extend: float
) -> np.ndarray:
    """生成指定轴向的边界区域布尔掩码"""
    lower, upper = center - extend, center + extend
    return (data[:, axis] >= lower) & (data[:, axis] <= upper)


def filter_vertical_outliers(boundary: np.ndarray) -> np.ndarray:
    """在高度方向过滤顶部和底部各5%的异常点"""
    if boundary.size == 0:
        return boundary
    z_vals = boundary[:, 2]
    lower, upper = np.quantile(z_vals, 0.05), np.quantile(z_vals, 0.95)
    return boundary[(z_vals >= lower) & (z_vals <= upper)]
