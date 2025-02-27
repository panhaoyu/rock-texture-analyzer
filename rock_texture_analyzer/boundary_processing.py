import numpy as np

from rock_texture_analyzer.clustering import process_clusters


def calculate_extended_bounds(left_center, right_center, front_center, back_center, extend_percent=0.1):
    """计算扩展后的边界范围"""
    extend_x = extend_percent * (right_center - left_center)
    extend_y = extend_percent * (back_center - front_center)
    return (
        extend_x, extend_y,
        left_center + extend_x,
        right_center - extend_x,
        front_center + extend_y,
        back_center - extend_y
    )


def filter_side_points(boundary_points, axis, center, extend, other_low, other_high):
    """筛选指定轴向的边界点"""
    axis_values = boundary_points[:, axis]
    mask = np.abs(axis_values - center) < extend
    side_points = boundary_points[mask]
    other_axis = 1 - axis  # 0对应x轴，1对应y轴
    return side_points[
        (side_points[:, other_axis] > other_low) &
        (side_points[:, other_axis] < other_high)
        ]


def calculate_final_boundaries(left_points, right_points, front_points, back_points, std_range=5):
    """计算基于统计的最终边界"""

    def calc_boundary(points, axis, is_positive):
        values = points[:, axis]
        offset = std_range * np.std(values)
        return np.mean(values) + offset if is_positive else np.mean(values) - offset

    return (
        calc_boundary(left_points, 0, True),  # left
        calc_boundary(right_points, 0, False),  # right
        calc_boundary(front_points, 1, True),  # front
        calc_boundary(back_points, 1, False)  # back
    )


def create_boundary_masks(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """创建边界区域和外围区域的掩码"""
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min_ext, x_max_ext = process_clusters(x_coords, extension_ratio=0.1)
    y_min_ext, y_max_ext = process_clusters(y_coords, extension_ratio=0.1)

    boundary_mask = (
            (x_coords >= x_min_ext) & (x_coords <= x_max_ext) &
            (y_coords >= y_min_ext) & (y_coords <= y_max_ext)
    )
    external_mask = (
            ((x_coords > x_max_ext) | (y_coords > y_max_ext)) &
            (x_coords >= x_min_ext)
    )
    return boundary_mask, external_mask


def generate_boundary_mask(data: np.ndarray, axis: int, center: float, extend: float) -> np.ndarray:
    """生成边界区域的布尔掩码"""
    lower = center - extend
    upper = center + extend
    return (data[:, axis] >= lower) & (data[:, axis] <= upper)


def filter_height(boundary: np.ndarray) -> np.ndarray:
    """在高度方向上过滤掉顶部和底部各5%的点"""
    if len(boundary) == 0:
        return boundary

    z_values = boundary[:, 2]
    lower = np.quantile(z_values, 0.05)
    upper = np.quantile(z_values, 0.95)
    return boundary[(z_values >= lower) & (z_values <= upper)]
