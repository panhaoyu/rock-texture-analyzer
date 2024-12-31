import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, cm
from sci_cache import sci_method_cache
from scipy.interpolate import griddata

from scripts.config import base_dir, project_name
from scripts.p6_仅保留顶面 import PointCloudProcessorP6


class PointCloudProcessorP7(PointCloudProcessorP6):
    grid_resolution = 0.1

    @property
    @sci_method_cache
    def p7_表面二维重建(self) -> np.ndarray:
        """
        将上表面的点云通过散点插值转换为 x, y 平面内的 [z, r, g, b] 四层矩阵。

        Args:
            resolution: (x_res, y_res)，表示插值后矩阵的大小，即 x 和 y 方向的分辨率。

        Returns:
            np.ndarray: 生成的矩阵，包含 [z, r, g, b] 四个层。
        """
        cloud = self.p6_仅保留顶面

        resolution: float = self.grid_resolution

        points = np.asarray(cloud.points)

        # 提取 x, y, z, r, g, b 数据
        x, y, z = points.T

        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        x_min += 0.2
        x_max -= 0.2
        y_min += 0.2
        y_max -= 0.2

        # 创建网格
        x_edge = np.arange(x_min, x_max, resolution)
        y_edge = np.arange(y_min, y_max, resolution)
        x_grid, y_grid = np.meshgrid(x_edge, y_edge)

        # 插值 z, r, g, b 数据
        arrays = []
        z_interp = griddata((x, y), z, (x_grid, y_grid), method='cubic')
        arrays.append(z_interp)
        colors = np.asarray(cloud.colors)
        if colors.size:
            r, g, b = colors.T
            r_interp = griddata((x, y), r, (x_grid, y_grid), method='cubic')
            g_interp = griddata((x, y), g, (x_grid, y_grid), method='cubic')
            b_interp = griddata((x, y), b, (x_grid, y_grid), method='cubic')
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

    def 绘制表面(self):
        interpolated_matrix = self.p7_表面二维重建

        # 绘制高程图
        elevation = interpolated_matrix[:, :, 0]
        norm = plt.Normalize(vmin=np.nanmin(elevation), vmax=np.nanmax(elevation))
        scalar_map = cm.ScalarMappable(cmap='jet', norm=norm)
        elevation = (scalar_map.to_rgba(elevation)[:, :, :3] * 255)
        elevation = np.uint8(elevation)
        Image.fromarray(elevation).save(self.ply_file.with_name('elevation.png'))

        if interpolated_matrix.shape[2] > 1:
            color = interpolated_matrix[:, :, 1:]
            for i in range(3):
                v = color[:, :, i]
                v_min = np.nanmin(v)
                v_max = np.nanmax(v)
                v = (v - v_min) / (v_max - v_min) * 255
                color[:, :, i] = v

            color = np.uint8(np.round(color))
            Image.fromarray(color).save(self.ply_file.with_name('surface.png'))

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制表面()


if __name__ == '__main__':
    PointCloudProcessorP7.main()
