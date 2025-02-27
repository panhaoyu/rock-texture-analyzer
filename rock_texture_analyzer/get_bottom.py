import numpy as np

from rock_texture_analyzer.k_means import get_centers_with_extension


def _create_boundary_masks(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """创建边界区域和外围区域的掩码"""
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min_ext, x_max_ext = get_centers_with_extension(x_coords)
    y_min_ext, y_max_ext = get_centers_with_extension(y_coords)

    boundary_mask = (
            (x_coords >= x_min_ext) & (x_coords <= x_max_ext) &
            (y_coords >= y_min_ext) & (y_coords <= y_max_ext)
    )
    external_mask = (
            ((x_coords > x_max_ext) | (y_coords > y_max_ext)) &
            (x_coords >= x_min_ext)
    )
    return boundary_mask, external_mask


def _are_points_empty(*point_arrays: np.ndarray) -> bool:
    """检查多个点数组是否至少有一个为空"""
    return any(len(points) == 0 for points in point_arrays)


def _should_flip_based_on_z(boundary_points: np.ndarray, external_points: np.ndarray) -> bool:
    """通过比较内外区域z中值判断是否需要翻转"""
    median_z_in = np.median(boundary_points[:, 2])
    median_z_out = np.median(external_points[:, 2])
    return median_z_out > median_z_in
