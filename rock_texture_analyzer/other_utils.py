import numpy as np


def should_flip_based_on_z(
        boundary_points: np.ndarray,
        external_points: np.ndarray
) -> bool:
    """通过比较内外区域Z轴中值判断是否需要翻转点云"""
    assert boundary_points.size and external_points.size, "输入数据不能为空"

    # 确保输入是二维数组
    boundary_z = boundary_points[:, 2] if boundary_points.ndim > 1 else boundary_points
    external_z = external_points[:, 2] if external_points.ndim > 1 else external_points

    # 比较中值高度
    return np.median(external_z) > np.median(boundary_z)
