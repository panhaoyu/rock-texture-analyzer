import numpy as np
from sci_cache import sci_method_cache
from scipy.ndimage import gaussian_filter

from scripts.config import project_name, base_dir
from scripts.p7_表面二维重建 import PointCloudProcessorP7


class PointCloudProcessorP8(PointCloudProcessorP7):
    @property
    @sci_method_cache
    def p8_一阶梯度幅值(self) -> np.ndarray:
        """
        计算一阶梯度的幅值，并将结果缓存起来。

        Returns:
            np.ndarray: 一阶梯度幅值的二维数组。
        """
        # 提取高度数据
        elevation = self.p7_表面二维重建[:, :, 0]

        # 计算一阶梯度
        grad_y, grad_x = np.gradient(elevation)

        # 对一阶梯度进行高斯平滑
        sigma = 2.0  # 可根据需要调整
        grad_x_smoothed = gaussian_filter(grad_x, sigma=sigma)
        grad_y_smoothed = gaussian_filter(grad_y, sigma=sigma)

        # 计算一阶梯度的幅值
        first_gradient_magnitude = np.hypot(grad_x_smoothed, grad_y_smoothed)

        return first_gradient_magnitude

    @property
    @sci_method_cache
    def p8_二阶梯度幅值(self) -> np.ndarray:
        """
        计算二阶梯度的幅值，并将结果缓存起来。

        Returns:
            np.ndarray: 二阶梯度幅值的二维数组。
        """
        # 提取高度数据
        elevation = self.p7_表面二维重建[:, :, 0]

        # 计算一阶梯度
        grad_y, grad_x = np.gradient(elevation)

        # 对一阶梯度进行高斯平滑
        sigma = 2.0  # 可根据需要调整
        grad_x_smoothed = gaussian_filter(grad_x, sigma=sigma)
        grad_y_smoothed = gaussian_filter(grad_y, sigma=sigma)

        # 计算二阶梯度
        d2f_dxx = np.gradient(grad_x_smoothed, axis=1)  # 二阶导数 x 方向
        d2f_dyy = np.gradient(grad_y_smoothed, axis=0)  # 二阶导数 y 方向

        # 计算二阶梯度的大小
        # second_gradient_magnitude = np.abs(d2f_dxx) + np.abs(d2f_dyy)
        second_gradient_magnitude = np.hypot(d2f_dxx, d2f_dyy)

        return second_gradient_magnitude

    @classmethod
    def main(cls):
        """
        主方法，执行表面绘制和梯度计算与保存。
        """
        self = cls(base_dir, project_name)

        # 获取插值后的表面数据
        interpolated_matrix = self.p7_表面二维重建  # 使用三次插值

        # 绘制并保存表面图像
        self.绘制表面(interpolated_matrix, name='surface')

        # 获取一阶梯度幅值
        first_gradient_magnitude = self.p8_一阶梯度幅值
        # 将一阶梯度幅值转换为 [z] 层的三维数组
        first_gradient_matrix = first_gradient_magnitude[:, :, np.newaxis]
        # 绘制并保存一阶梯度图像
        self.绘制表面(first_gradient_matrix, name='first_gradient')

        # 获取二阶梯度幅值
        second_gradient_magnitude = self.p8_二阶梯度幅值
        # 将二阶梯度幅值转换为 [z] 层的三维数组
        second_gradient_matrix = second_gradient_magnitude[:, :, np.newaxis]
        # 绘制并保存二阶梯度图像
        self.绘制表面(second_gradient_matrix, name='second_gradient')


if __name__ == '__main__':
    PointCloudProcessorP8.main()
