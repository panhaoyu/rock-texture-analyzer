from typing import Tuple

import numpy as np

from rock_texture_analyzer.clustering import process_clusters

def compute_extended_bounds(
    start_center: float,
    end_center: float,
    extend_percent: float = 0.1
) -> tuple[float, float, float]:
    """计算单方向扩展边界，返回扩展量和扩展后的起止边界"""
    extend = extend_percent * (end_center - start_center)
    return extend, start_center + extend, end_center - extend


def compute_boundary(
        boundary_points: np.ndarray,
        axis: int,
        center: float,
        extend: float,
        other_low: float,
        other_high: float,
        is_positive: bool,
        std_range: int = 5
) -> float:
    """计算单方向的统计边界"""
    # 筛选指定轴向的边界点
    axis_vals = boundary_points[:, axis]
    other_axis = 1 - axis
    mask = (np.abs(axis_vals - center) < extend)
    mask &= (boundary_points[:, other_axis] > other_low)
    mask &= (boundary_points[:, other_axis] < other_high)
    filtered_points = boundary_points[mask]

    # 计算统计边界
    values = filtered_points[:, axis]
    offset = std_range * np.std(values)
    return (np.mean(values) + offset) if is_positive else (np.mean(values) - offset)


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
