import numpy as np
import open3d
from scipy.interpolate import griddata
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from p3_表面显微镜扫描数据处理.config import debug
from rock_texture_analyzer.base import BaseProcessor
from rock_texture_analyzer.utils.get_two_peaks import get_two_main_value_filtered, ValueDetectionError


def find_valid_clusters(point_x, point_y, thresholds):
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


def are_points_empty(*point_arrays: np.ndarray) -> bool:
    """检查多个点数组是否至少有一个为空"""
    return any(len(points) == 0 for points in point_arrays)


def should_flip_based_on_z(boundary_points: np.ndarray, external_points: np.ndarray) -> bool:
    """通过比较内外区域z中值判断是否需要翻转"""
    median_z_in = np.median(boundary_points[:, 2])
    median_z_out = np.median(external_points[:, 2])
    return median_z_out > median_z_in


def get_centers_with_extension(
        data: np.ndarray,
        n_clusters: int = 2,
        extension_ratio: float = 0.1
) -> tuple[float, float]:
    """获取数据聚类中心点并扩展范围"""
    kmeans: KMeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data.reshape(-1, 1))
    centers = sorted(c for c in kmeans.cluster_centers_.flatten())
    min_val, max_val = min(centers), max(centers)
    extent = (max_val - min_val) * (1 + extension_ratio)
    return min_val - extent, max_val + extent


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


def surface_interpolate_2d(cloud: open3d.geometry.PointCloud, resolution: float, method: str) -> np.ndarray:
    """执行二维插值运算，返回[z, r, g, b]四层矩阵"""
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    _debug_ratio = 100

    if debug:
        points = points[::_debug_ratio]
        colors = colors[::_debug_ratio]

    x, y, z = points.T
    buffer = 0.2  # 边界缓冲区大小

    # 计算有效范围并确保x_min < x_max
    x_min, x_max = np.min(x) - buffer, np.max(x) + buffer
    y_min, y_max = np.min(y) - buffer, np.max(y) + buffer

    # 生成等间距网格（使用linspace确保覆盖完整范围）
    x_grid = np.arange(x_min, x_max + resolution, resolution)
    y_grid = np.arange(y_min, y_max + resolution, resolution)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    # Z值插值
    z_interp = griddata((x, y), z, (x_grid, y_grid), method=method)
    arrays = [z_interp]

    # 颜色插值处理
    if colors.size:
        r, g, b = colors.T
        for channel in [r, g, b]:
            interp = griddata((x, y), channel, (x_grid, y_grid), method=method)
            # 限制颜色值在[0,1]范围内
            arrays.append(np.clip(interp, 0, 1))
    else:
        # 创建默认颜色通道（全零）
        null_channel = np.zeros_like(z_interp)
        arrays.extend([null_channel] * 3)

    return np.stack(arrays, axis=-1)
