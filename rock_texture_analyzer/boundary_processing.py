from typing import Tuple

import numpy as np

from rock_texture_analyzer.clustering import process_clusters


def compute_extended_bounds(
        left_center: float,
        right_center: float,
        front_center: float,
        back_center: float,
        extend_percent: float = 0.1
) -> tuple[float, float, float, float, float, float]:
    """计算扩展后的边界范围，返回扩展量和扩展后的各方向边界值"""
    extend_x = extend_percent * (right_center - left_center)
    extend_y = extend_percent * (back_center - front_center)
    return (
        extend_x, extend_y,
        left_center + extend_x,
        right_center - extend_x,
        front_center + extend_y,
        back_center - extend_y
    )


def filter_points_by_axis(
        boundary_points: np.ndarray,
        axis: int,
        center: float,
        extend: float,
        other_low: float,
        other_high: float
) -> np.ndarray:
    """筛选指定轴向的边界点，返回过滤后的点集"""
    axis_vals = boundary_points[:, axis]
    other_axis = 1 - axis
    mask = (np.abs(axis_vals - center) < extend)
    mask &= (boundary_points[:, other_axis] > other_low)
    mask &= (boundary_points[:, other_axis] < other_high)
    return boundary_points[mask]


def compute_statistical_boundaries(
        left_points: np.ndarray,
        right_points: np.ndarray,
        front_points: np.ndarray,
        back_points: np.ndarray,
        std_range: int = 5
) -> tuple[float, float, float, float]:
    """计算基于统计方法的最终边界"""

    def calc_boundary(points: np.ndarray, axis: int, is_positive: bool) -> float:
        """计算单边统计边界"""
        values = points[:, axis]
        offset = std_range * np.std(values)
        return (np.mean(values) + offset) if is_positive else (np.mean(values) - offset)

    return (
        calc_boundary(left_points, 0, True),
        calc_boundary(right_points, 0, False),
        calc_boundary(front_points, 1, True),
        calc_boundary(back_points, 1, False)
    )


def create_boundary_masks(
        points: np.ndarray,
        extension_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """创建边界区域和外围区域的布尔掩码"""
    x_coords, y_coords = points[:, 0], points[:, 1]
    x_min, x_max = process_clusters(x_coords, extension_ratio=extension_ratio)
    y_min, y_max = process_clusters(y_coords, extension_ratio=extension_ratio)
    boundary_mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
    external_mask = (x_coords > x_max) | (y_coords > y_max) & (x_coords >= x_min)
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
