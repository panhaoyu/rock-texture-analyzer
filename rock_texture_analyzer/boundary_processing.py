from typing import Tuple

import numpy as np

from rock_texture_analyzer.clustering import process_clusters, find_two_peaks, ValueDetectionError, find_single_peak


def get_boundaries(points: np.ndarray) -> tuple[float, float, float, float, float, float]:
    point_z = points[:, 2]
    thresholds = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    try:
        bottom, top = find_two_peaks(point_z, thresholds)
    except ValueDetectionError:
        bottom, top = np.min(point_z), find_single_peak(point_z, thresholds)
    z_selector = (point_z > (bottom + (top - bottom) * 0.1)) & (point_z < (top - (top - bottom) * 0.4))
    boundary_points = points[z_selector]
    point_x, point_y = boundary_points[:, 0], boundary_points[:, 1]
    left_center, right_center = find_two_peaks(point_x, thresholds)
    front_center, back_center = find_two_peaks(point_y, thresholds)
    assert back_center > front_center and right_center > left_center

    # todo 拆分为两行
    extend_x, extend_y = 0.1 * (right_center - left_center), 0.1 * (back_center - front_center)

    # todo 各个mask都分别独立定义，例如，left_mask
    # todo boundary_points[:, 0] 与 1 分别独立定义为变量，以简化后面的写法。
    # todo 避免使用多行表达式，可以多定义一些变量
    mask = (np.abs(boundary_points[:, 0] - left_center) < extend_x) & (
            boundary_points[:, 1] > (front_center + extend_y)) & (boundary_points[:, 1] < (back_center - extend_y))
    left = np.mean(boundary_points[mask][:, 0]) + 5 * np.std(boundary_points[mask][:, 0])
    mask = (np.abs(boundary_points[:, 0] - right_center) < extend_x) & (
            boundary_points[:, 1] > (front_center + extend_y)) & (boundary_points[:, 1] < (back_center - extend_y))
    right = np.mean(boundary_points[mask][:, 0]) - 5 * np.std(boundary_points[mask][:, 0])
    mask = (np.abs(boundary_points[:, 1] - front_center) < extend_y) & (
            boundary_points[:, 0] > (left_center + extend_x)) & (boundary_points[:, 0] < (right_center - extend_x))
    front = np.mean(boundary_points[mask][:, 1]) + 5 * np.std(boundary_points[mask][:, 1])
    mask = (np.abs(boundary_points[:, 1] - back_center) < extend_y) & (
            boundary_points[:, 0] > (left_center + extend_x)) & (boundary_points[:, 0] < (right_center - extend_x))
    back = np.mean(boundary_points[mask][:, 1]) - 5 * np.std(boundary_points[mask][:, 1])
    return left, right, front, back, bottom, top


def create_boundary_masks(
        points: np.ndarray,
        extension_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """创建边界区域和外围区域的布尔掩码"""
    x_coords, y_coords = points[:, 0], points[:, 1]

    # 计算扩展后的边界区域
    x_min, x_max = process_clusters(x_coords, extension_ratio=extension_ratio)
    y_min, y_max = process_clusters(y_coords, extension_ratio=extension_ratio)

    # 生成边界区域掩码
    boundary_mask = (
            (x_coords >= x_min) & (x_coords <= x_max) &
            (y_coords >= y_min) & (y_coords <= y_max)
    )

    # 生成外围区域掩码（边界之外的区域）
    external_mask = (
            (x_coords < x_min) | (x_coords > x_max) |
            (y_coords < y_min) | (y_coords > y_max)
    )

    return boundary_mask, external_mask


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
