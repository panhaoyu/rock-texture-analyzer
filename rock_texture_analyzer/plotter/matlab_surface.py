import numpy as np


def matlab_surface(array: np.ndarray):
    """
    使用 MATLAB 绘制三维表面高程图。

    Args:
        array (np.ndarray): 输入的三维矩阵，形状为 (M, N, 1) 或 (M, N, 4)。
    """
    # 只取第一层作为高程数据
    elevation = array[:, :, 0]
    elevation_min, elevation_max = np.nanquantile(elevation, 0.01), np.nanquantile(elevation, 0.99)
    elevation[elevation < elevation_min] = elevation_min
    elevation[elevation > elevation_max] = elevation_max

    # 获取矩阵大小
    M, N = elevation.shape

    resolution = self.grid_resolution

    # 生成x和y坐标，确保长度为 N 和 M
    x_edge = np.linspace(0, (N - 1) * resolution, N)
    y_edge = np.linspace(0, (M - 1) * resolution, M)
    x_grid, y_grid = np.meshgrid(x_edge, y_edge)

    # 确认网格和Z的形状一致
    if x_grid.shape != elevation.shape or y_grid.shape != elevation.shape:
        raise ValueError(f"网格和高程数据的形状不匹配: X={x_grid.shape}, Y={y_grid.shape}, Z={elevation.shape}")

    eng = matlab.engine.start_matlab()

    X = matlab.double(x_grid.tolist())
    Y = matlab.double(y_grid.tolist())
    Z = matlab.double(elevation.tolist())

    # 调用 MATLAB 绘图函数
    self.print_safe("在 MATLAB 中绘制三维表面...")
    eng.figure(nargout=0)
    eng.mesh(X, Y, Z, nargout=0)
    eng.grid(nargout=0)

    output_dir = self.ply_file.with_name('images')
    output_dir.mkdir(parents=True, exist_ok=True)
    matlab_plot_path = str(output_dir.joinpath('matlab_surface_plot.png'))
    eng.savefig(matlab_plot_path, nargout=0)
    self.print_safe(f"MATLAB 图像已保存到 {matlab_plot_path}")

    # 显示 MATLAB 图形窗口
    eng.show(nargout=0)
