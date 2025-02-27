import numpy as np
import open3d
from scipy.interpolate import griddata

from p3_表面显微镜扫描数据处理.config import debug

_debug_ratio = 100


def surface_interpolate_2d(cloud: open3d.geometry.PointCloud, resolution: float, method: str) -> np.ndarray:
    """执行二维插值运算，返回[z, r, g, b]四层矩阵"""
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)

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
