import numpy as np


def should_flip_based_on_z(boundary_points: np.ndarray, external_points: np.ndarray) -> bool:
    """通过比较内外区域z中值判断是否需要翻转"""
    median_z_in = np.median(boundary_points[:, 2])
    median_z_out = np.median(external_points[:, 2])
    return median_z_out > median_z_in
