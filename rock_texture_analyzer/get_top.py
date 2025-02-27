import numpy as np

from rock_texture_analyzer.utils.get_two_peaks import get_two_main_value_filtered, ValueDetectionError


def _find_valid_clusters(point_x, point_y, thresholds):
    """通过尝试不同阈值获取有效聚类中心"""
    for threshold in thresholds:
        try:
            return (
                *get_two_main_value_filtered(point_x, threshold),
                *get_two_main_value_filtered(point_y, threshold)
            )
        except ValueDetectionError:
            continue
    raise ValueDetectionError(f"无法找到有效阈值，尝试了所有阈值: {thresholds}")


def _calculate_extended_bounds(left_center, right_center, front_center, back_center, extend_percent=0.1):
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


def _filter_side_points(boundary_points, axis, center, extend, other_low, other_high):
    """筛选指定轴向的边界点"""
    axis_values = boundary_points[:, axis]
    mask = np.abs(axis_values - center) < extend
    side_points = boundary_points[mask]
    other_axis = 1 - axis  # 0对应x轴，1对应y轴
    return side_points[
        (side_points[:, other_axis] > other_low) &
        (side_points[:, other_axis] < other_high)
        ]


def _calculate_final_boundaries(left_points, right_points, front_points, back_points, std_range=5):
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
