import ezdxf
import matlab.engine
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, cm
from sci_cache import sci_method_cache
from scipy.interpolate import griddata

from p2_点云数据处理.config import base_dir, project_name
from p2_点云数据处理.p6_仅保留顶面 import PointCloudProcessorP6


class PointCloudProcessorP7(PointCloudProcessorP6):


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
        """
        # 提取并处理高程数据
        resolution_mm = 2.0
        skip_ratio = int(resolution_mm / self.grid_resolution)
        elevation = array[::skip_ratio, ::skip_ratio, 0]
        elevation_min, elevation_max = np.nanquantile(elevation, 0.01), np.nanquantile(elevation, 0.99)
        elevation = np.clip(elevation, elevation_min, elevation_max)

        # 高度放缩系数
        scale_z = 1.0
        avg_z = np.nanmean(elevation)
        elevation = (elevation - avg_z) * scale_z + avg_z

        M, N = elevation.shape
        x_edge = np.linspace(0, (N - 1) * resolution_mm, N)
        y_edge = np.linspace(0, (M - 1) * resolution_mm, M)

        # 创建 DXF 文档
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        # Collect vertices and create a vertex map
        vertices = []
        vertex_map = {}
        idx = 0
        for i in range(M):
            for j in range(N):
                z_val = elevation[i, j]
                if not np.isnan(z_val):
                    x = x_edge[j]
                    y = y_edge[i]
                    vertices.append((x, y, z_val))
                    vertex_map[(i, j)] = idx
                    idx += 1

        # Add vertices and faces using edit_data
        mesh = msp.add_mesh()
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices.extend(vertices)
            for i in range(M - 1):
                for j in range(N - 1):
                    if ((i, j) in vertex_map and (i, j + 1) in vertex_map and
                            (i + 1, j) in vertex_map and (i + 1, j + 1) in vertex_map):
                        v0 = vertex_map[(i, j)]
                        v1 = vertex_map[(i, j + 1)]
                        v2 = vertex_map[(i + 1, j)]
                        v3 = vertex_map[(i + 1, j + 1)]
                        mesh_data.faces.append([v0, v1, v2])
                        mesh_data.faces.append([v2, v1, v3])

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
        obj.绘制表面(obj.p7_表面二维重建)
        # obj.绘制三维表面_matlab(obj.p7_表面二维重建)
        # obj.绘制表面_导出到AutoCAD(obj.p7_表面二维重建)


if __name__ == '__main__':
    PointCloudProcessorP7.main()
