import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from rock_texture_analyzer.base import BaseProcessor


def cluster_axis(data: np.ndarray, axis: int) -> tuple:
    """对指定轴进行K-Means聚类并返回排序后的聚类中心"""
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data[:, axis].reshape(-1, 1))
    centers = sorted(kmeans.cluster_centers_.flatten())
    return centers[0], centers[1]


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


def compute_rotated_std(R: np.ndarray, boundaries: list) -> float:
    """计算旋转后各边界的标准差之和"""
    rotated = [b.dot(R.T) for b in boundaries]
    std_values = [
        np.std(rotated[0][:, 0]),  # 左侧X轴
        np.std(rotated[1][:, 0]),  # 右侧X轴
        np.std(rotated[2][:, 1]),  # 前侧Y轴
        np.std(rotated[3][:, 1])  # 后侧Y轴
    ]
    return sum(std_values)


def optimize_angles(initial_angles: list, boundaries: list, bounds: tuple) -> dict:
    """执行优化过程并返回结果"""

    def objective(angles_deg):
        R = create_rotation_matrix(angles_deg)
        return compute_rotated_std(R, boundaries)

    result = minimize(
        objective,
        initial_angles,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-8, 'maxiter': 1000}
    )

    return {
        'success': result.success,
        'angles': result.x,
        'value': result.fun
    }


def least_squares_adjustment_direction(points: np.ndarray) -> np.ndarray:
    # 1. 对X/Y轴进行聚类
    xmin, xmax = cluster_axis(points, 0)
    ymin, ymax = cluster_axis(points, 1)

    # 2. 计算扩展范围
    extend_x = 0.1 * (xmax - xmin)
    extend_y = 0.1 * (ymax - ymin)

    # 3. 提取各边界点
    boundaries = [
        points[generate_boundary_mask(points, 0, xmin, extend_x)],  # 左侧
        points[generate_boundary_mask(points, 0, xmax, extend_x)],  # 右侧
        points[generate_boundary_mask(points, 1, ymin, extend_y)],  # 前侧
        points[generate_boundary_mask(points, 1, ymax, extend_y)]  # 后侧
    ]

    # 4. 高度过滤
    boundaries = [filter_height(b) for b in boundaries]
    if any(len(b) == 0 for b in boundaries):
        BaseProcessor.print_safe("某些边界在高度过滤后没有剩余的点。")
        return

    # 5. 优化旋转角度
    optimization_result = optimize_angles(
        initial_angles=[0.0, 0.0, 0.0],
        boundaries=boundaries,
        bounds=[(-10, 10)] * 3
    )

    # 6. 处理优化结果
    if optimization_result['success']:
        best_angles = optimization_result['angles']
        BaseProcessor.print_safe(
            f"最佳旋转角度 (α, β, γ): {best_angles} 度, 总标准差: {optimization_result['value']:.6f}")
    else:
        BaseProcessor.print_safe("优化未收敛，使用初始角度。")
        best_angles = [0.0, 0.0, 0.0]

    # 7. 生成最终旋转矩阵
    return create_rotation_matrix(best_angles)
