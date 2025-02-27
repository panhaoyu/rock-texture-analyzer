import numpy as np
import open3d
from scipy.interpolate import griddata


def surface_interpolate_2d(cloud: open3d.geometry.PointCloud, resolution: float, method: str) -> np.ndarray:
    """执行二维插值运算，返回[z, r, g, b]四层矩阵"""
    # todo 先把 points colors 合并为一个大矩阵，就是先做stack，然后再根据维度来判断是否要读取
    points = np.asarray(cloud.points)
    x, y, z = points.T
    x_min, x_max = np.min(x) + 0.2, np.max(x) - 0.2
    y_min, y_max = np.min(y) + 0.2, np.max(y) - 0.2

    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, resolution),
                                 np.arange(y_min, y_max, resolution))
    arrays = [griddata((x, y), z, (x_grid, y_grid), method=method)]

    if (colors := np.asarray(cloud.colors)).size:
        r, g, b = colors.T
        arrays.extend(griddata((x, y), c, (x_grid, y_grid), method=method)
                      for c in [r, g, b])

    return np.stack(arrays, axis=-1)
