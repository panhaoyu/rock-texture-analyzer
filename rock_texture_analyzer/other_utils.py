import numpy as np

from rock_texture_analyzer.clustering import process_clusters


def should_flip_based_on_z(points: np.ndarray) -> bool:
    """通过比较内外区域Z轴中值判断是否需要翻转点云"""
    extension_ratio = 0.1
    x, y = points[:, 0], points[:, 1]
    x_min, x_max = process_clusters(x, extension_ratio=extension_ratio)
    y_min, y_max = process_clusters(y, extension_ratio=extension_ratio)

    boundary_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    external_mask = ~boundary_mask

    boundary_z = points[boundary_mask, 2]
    external_z = points[external_mask, 2]
    assert boundary_z.size and external_z.size, "输入数据不能为空"

    return np.median(external_z) > np.median(boundary_z)


def compute_rotation_matrix(plane_normal: np.ndarray, target_normal: np.ndarray) -> np.ndarray:
    """计算将平面法向量旋转到目标法向量的旋转矩阵"""
    v = np.cross(plane_normal, target_normal)
    s, c = np.linalg.norm(v), np.dot(plane_normal, target_normal)
    if s < 1e-6:
        return np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))


def create_rotation_matrix(angles_deg: tuple) -> np.ndarray:
    """根据给定的欧拉角创建旋转矩阵（XYZ顺序）"""
    alpha, beta, gamma = np.radians(angles_deg)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    R_z = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x
