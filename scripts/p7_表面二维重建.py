import ezdxf
import matlab.engine
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, cm
from sci_cache import sci_method_cache
from scipy.interpolate import griddata

from scripts.config import base_dir, project_name
from scripts.p6_仅保留顶面 import PointCloudProcessorP6


class PointCloudProcessorP7(PointCloudProcessorP6):
    grid_resolution = 0.1

    def _interpolate_surface(self, method: str) -> np.ndarray:
        """
        使用指定的插值方法将点云数据插值到二维网格上。

        Args:
            method (str): 插值方法，例如 'nearest', 'linear', 'cubic'。

        Returns:
            np.ndarray: 插值后的 [z, r, g, b] 四层矩阵。
        """
        cloud = self.p6_仅保留顶面

        resolution: float = self.grid_resolution

        points = np.asarray(cloud.points)

        # 提取 x, y, z 数据
        x, y, z = points.T

        x_min = np.min(x) + 0.2
        x_max = np.max(x) - 0.2
        y_min = np.min(y) + 0.2
        y_max = np.max(y) - 0.2

        # 创建网格
        x_edge = np.arange(x_min, x_max, resolution)
        y_edge = np.arange(y_min, y_max, resolution)
        x_grid, y_grid = np.meshgrid(x_edge, y_edge)

        # 插值 z, r, g, b 数据
        arrays = []
        z_interp = griddata((x, y), z, (x_grid, y_grid), method=method)
        arrays.append(z_interp)
        colors = np.asarray(cloud.colors)
        if colors.size:
            r, g, b = colors.T
            r_interp = griddata((x, y), r, (x_grid, y_grid), method=method)
            g_interp = griddata((x, y), g, (x_grid, y_grid), method=method)
            b_interp = griddata((x, y), b, (x_grid, y_grid), method=method)
            arrays.extend([r_interp, g_interp, b_interp])

        # 生成 [z, r, g, b] 四层矩阵
        interpolated_matrix = np.stack(arrays, axis=-1)

        layer_names = ['z', 'r', 'g', 'b']
        num_layers = interpolated_matrix.shape[-1]
        for i in range(num_layers):
            layer = interpolated_matrix[:, :, i]
            total = layer.size
            nan_count = np.isnan(layer).sum()
            nan_percentage = (nan_count / total) * 100
            layer_name = layer_names[i] if i < len(layer_names) else f'layer_{i}'
            print(f"Layer '{layer_name}':")
            print(f"  总元素数量 (Total elements) = {total}")
            print(f"  NaN 数量 (NaN count) = {nan_count}")
            print(f"  NaN 占比 (NaN percentage) = {nan_percentage:.2f}%\n")

        return interpolated_matrix

    @property
    @sci_method_cache
    def p7_表面二维重建_最近邻插值(self) -> np.ndarray:
        """
        使用最近邻插值方法进行表面二维重建。

        Returns:
            np.ndarray: 插值后的矩阵。
        """
        return self._interpolate_surface(method='nearest')

    @property
    @sci_method_cache
    def p7_表面二维重建_线性插值(self) -> np.ndarray:
        """
        使用线性插值方法进行表面二维重建。

        Returns:
            np.ndarray: 插值后的矩阵。
        """
        return self._interpolate_surface(method='linear')

    @property
    @sci_method_cache
    def p7_表面二维重建_三次插值(self) -> np.ndarray:
        """
        使用三次插值方法进行表面二维重建。

        Returns:
            np.ndarray: 插值后的矩阵。
        """
        return self._interpolate_surface(method='cubic')

    @property
    @sci_method_cache
    def p7_表面二维重建(self) -> np.ndarray:
        """
        返回使用三次插值方法进行的表面二维重建结果。

        Returns:
            np.ndarray: 插值后的矩阵。
        """
        return self.p7_表面二维重建_三次插值

    def 绘制表面(self, interpolated_matrix: np.ndarray, name: str = 'output'):
        """
        绘制表面高程图和颜色图，保存后根据是否存在颜色图像进行显示。
        如果存在颜色图像，则将高程图和颜色图左右合并并显示。
        如果不存在颜色图像，则仅显示高程图像。

        Args:
            interpolated_matrix (np.ndarray): 预先计算好的插值结果矩阵。
            name (str): 名称的前缀。
        """
        # 绘制高程图
        elevation = interpolated_matrix[:, :, 0]
        norm = plt.Normalize(
            vmin=float(np.nanquantile(elevation, 0.01)),
            vmax=float(np.nanquantile(elevation, 0.99)),
        )
        scalar_map = cm.ScalarMappable(cmap='jet', norm=norm)
        elevation_rgba = scalar_map.to_rgba(elevation)
        elevation_rgb = (elevation_rgba[:, :, :3] * 255).astype(np.uint8)
        elevation_image = Image.fromarray(elevation_rgb)
        output_dir = self.ply_file.with_name('images')
        output_dir.mkdir(parents=True, exist_ok=True)
        elevation_path = output_dir.joinpath(f'{name}-elevation.png')
        elevation_image.save(elevation_path)

        surface_image = None
        if interpolated_matrix.shape[2] > 1:
            # 处理颜色数据
            color = interpolated_matrix[:, :, 1:].copy()
            for i in range(3):
                v = color[:, :, i]
                v_min = np.nanquantile(v, 0.01)
                v_max = np.nanquantile(v, 0.99)
                if v_max - v_min == 0:
                    normalized_v = np.zeros_like(v)
                else:
                    normalized_v = (v - v_min) / (v_max - v_min) * 255
                color[:, :, i] = normalized_v

            color_uint8 = np.clip(np.round(color), 0, 255).astype(np.uint8)
            surface_image = Image.fromarray(color_uint8)
            surface_path = output_dir.joinpath(f'{name}-surface.png')
            surface_image.save(surface_path)

        # 显示图像
        if surface_image:
            # 确保两张图像的尺寸相同
            if elevation_image.size != surface_image.size:
                raise ValueError("高程图和颜色图的尺寸不一致，无法合并显示。")

            # 创建一个新的空白图像，宽度为两张图像的总和，高度相同
            combined_width = elevation_image.width + surface_image.width
            combined_height = elevation_image.height
            combined_image = Image.new('RGB', (combined_width, combined_height))

            # 将高程图粘贴到左侧
            combined_image.paste(elevation_image, (0, 0))

            # 将颜色图粘贴到右侧
            combined_image.paste(surface_image, (elevation_image.width, 0))

            # 显示合并后的图像
            combined_image.show()
        else:
            # 仅显示高程图像
            elevation_image.show()

    def 绘制三维表面_matlab(self, array: np.ndarray):
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
        print("在 MATLAB 中绘制三维表面...")
        eng.figure(nargout=0)
        eng.mesh(X, Y, Z, nargout=0)
        eng.grid(nargout=0)

        output_dir = self.ply_file.with_name('images')
        output_dir.mkdir(parents=True, exist_ok=True)
        matlab_plot_path = str(output_dir.joinpath('matlab_surface_plot.png'))
        eng.savefig(matlab_plot_path, nargout=0)
        print(f"MATLAB 图像已保存到 {matlab_plot_path}")

        # 显示 MATLAB 图形窗口
        eng.show(nargout=0)

    def 绘制表面_导出到AutoCAD(self, array: np.ndarray):
        """
        将插值后的高程数据导出为 AutoCAD 支持的 DXF 文件。

        Args:
            array (np.ndarray): 插值后的 [z, r, g, b] 矩阵。
            filename (str): 导出 DXF 文件的名称。
        """
        # 创建一个新的 DXF 文档
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        # 提取高程数据
        z = array[:, :, 0]

        num_rows, num_cols = z.shape

        # 获取点云的 x 和 y 范围
        points = np.asarray(self.p6_仅保留顶面.points)
        x_min = np.min(points[:, 0]) + 0.2
        y_min = np.min(points[:, 1]) + 0.2

        # 遍历网格，添加 3D 点到 DXF
        for i in range(num_rows):
            for j in range(num_cols):
                elevation = z[i, j]
                if not np.isnan(elevation):
                    x = x_min + j * self.grid_resolution
                    y = y_min + i * self.grid_resolution
                    msp.add_point((x, y, elevation))

        # 保存 DXF 文件
        output_dir = self.ply_file.with_name('autocad_exports')
        output_dir.mkdir(parents=True, exist_ok=True)
        dxf_path = output_dir.joinpath('elevation.dxf')
        doc.saveas(str(dxf_path))
        print(f"高程数据已成功导出到 {dxf_path}")

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        # 使用不同的缓存属性绘制表面
        # obj.绘制表面(obj.p7_表面二维重建_三次插值)  # 使用三次插值
        # obj.绘制表面(obj.p7_表面二维重建_线性插值)  # 使用线性插值
        # obj.绘制表面(obj.p7_表面二维重建_最近邻插值)  # 使用最近邻插值
        # obj.绘制表面(obj.p7_表面二维重建)
        # obj.绘制三维表面_matlab(obj.p7_表面二维重建)
        obj.绘制表面_导出到AutoCAD(obj.p7_表面二维重建)

if __name__ == '__main__':
    PointCloudProcessorP7.main()
