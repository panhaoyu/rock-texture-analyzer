import numpy as np
from PIL import Image
from matplotlib import cm
from scipy.ndimage import gaussian_filter

from scripts.config import base_dir, project_name
from scripts.p7_表面二维重建 import PointCloudProcessorP7


class PointCloudProcessorP8(PointCloudProcessorP7):
    def 计算并保存二阶梯度图像(self, interpolated_matrix: np.ndarray, name: str = 'second_gradient'):
        """
        计算二阶梯度，并将其幅值映射为彩色图像后保存为PNG图像。

        Args:
            interpolated_matrix (np.ndarray): 预先计算好的插值结果矩阵。
            name (str): 输出文件的名称前缀。
        """
        # 提取高度数据
        elevation = interpolated_matrix[:, :, 0]

        # 计算一阶梯度
        grad_y, grad_x = np.gradient(elevation)

        # 对一阶梯度进行高斯平滑
        sigma = 1.0  # 可根据需要调整
        grad_x_smoothed = gaussian_filter(grad_x, sigma=sigma)
        grad_y_smoothed = gaussian_filter(grad_y, sigma=sigma)

        # 计算二阶梯度
        d2f_dxx = np.gradient(grad_x_smoothed, axis=1)  # 二阶导数 x 方向
        d2f_dyy = np.gradient(grad_y_smoothed, axis=0)  # 二阶导数 y 方向
        d2f_dxy = np.gradient(grad_x_smoothed, axis=0)  # 混合二阶导数

        # 计算二阶梯度的大小
        second_gradient_magnitude = np.abs(d2f_dxx) + np.abs(d2f_dyy)
        # 或者使用其他计算方式，例如：
        # second_gradient_magnitude = np.sqrt(d2f_dxx**2 + d2f_dyy**2)

        # 归一化二阶梯度幅值到 [0, 1]
        min_val = np.nanmin(second_gradient_magnitude)
        max_val = np.nanmax(second_gradient_magnitude)
        norm = (second_gradient_magnitude - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(
            second_gradient_magnitude)
        norm = np.nan_to_num(norm)  # 将 NaN 转换为 0

        # 使用 'jet' 颜色映射
        cmap = cm.get_cmap('jet')
        gradient_rgba = cmap(norm)  # 返回 RGBA
        gradient_rgb = (gradient_rgba[:, :, :3] * 255).astype(np.uint8)  # 转换为 RGB 并缩放到 [0, 255]

        # 创建 PIL 图像
        gradient_image = Image.fromarray(gradient_rgb)

        # 保存梯度图像
        output_dir = self.ply_file.with_name('images')
        output_dir.mkdir(parents=True, exist_ok=True)
        gradient_path = output_dir.joinpath(f'{name}-second_gradient.png')
        gradient_image.save(gradient_path)
        print(f"二阶梯度幅值图像已保存到 {gradient_path}")

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        interpolated_matrix = obj.p7_表面二维重建  # 使用三次插值
        obj.绘制表面(interpolated_matrix, name='surface')

        # 计算并保存二阶梯度图像
        obj.计算并保存二阶梯度图像(interpolated_matrix, name='second_gradient')


if __name__ == '__main__':
    PointCloudProcessorP8.main()
