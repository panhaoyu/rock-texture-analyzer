import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from more_itertools import only

import batch_processor.processors.base
from batch_processor.batch_processor import SerialProcess
from batch_processor.processors.base import ManuallyProcessRequiredException
from batch_processor.processors.png import mark_as_png


class s25010801_花岗岩剪切前的断面光学扫描(SerialProcess):
    v6_转换为凸多边形的检测长度_像素 = 100
    v9_不参与拉伸系数计算的边界宽度_像素 = 100
    v12_不参与垂直拉伸系数计算的边界高度_像素 = 100
    v15_图像黑边 = 10
    v15_最终图像宽度_像素 = 4000
    v15_最终图像高度_像素 = 4000

    @mark_as_png
    def f1_原始数据(self):
        raise ManuallyProcessRequiredException

    @mark_as_png
    def f2_上下扩展(self):
        array = np.asarray(self.f1_原始数据)
        extended_array = np.pad(array, ((1000, 1000), (0, 0), (0, 0)), mode='edge')
        return Image.fromarray(extended_array)

    @mark_as_png
    def f3_删除非主体的像素(self):
        raise ManuallyProcessRequiredException

    @mark_as_png
    def f4_非透明部分的mask(self):
        array = np.asarray(self.f3_删除非主体的像素)
        mask = (array[..., 3] > 0).astype(np.uint8) * 255
        return Image.fromarray(mask, 'L')

    @mark_as_png
    def f5_提取最大的区域(self):
        array = np.asarray(self.f4_非透明部分的mask)
        contours = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        largest = max(contours, key=cv2.contourArea)
        convex = np.zeros_like(array)
        cv2.drawContours(convex, [largest], -1, (255,), thickness=cv2.FILLED)
        return Image.fromarray(convex, 'L')

    @mark_as_png
    def f6_转换为凸多边形(self):
        array = np.asarray(self.f5_提取最大的区域)
        contour = only(cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        convex = np.zeros_like(array)
        M = cv2.moments(contour)
        window_size = self.v6_转换为凸多边形的检测长度_像素
        center = np.array([[[int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]]])
        contour = np.concatenate([contour, contour[:window_size]])
        for i in range(len(contour) - window_size + 1):
            window = contour[i:i + window_size]
            window_with_center = np.concatenate([window, center])
            hull = cv2.convexHull(window_with_center.reshape(-1, 1, 2))
            cv2.drawContours(convex, [hull], -1, (255,), thickness=cv2.FILLED)
        return Image.fromarray(convex, 'L')

    @mark_as_png
    def f7_显示识别效果(self):
        f2_array = np.asarray(self.f2_上下扩展)
        f6_array = np.asarray(self.f6_转换为凸多边形)
        f2_array[..., 0] = np.where(f6_array == 255, 255, f2_array[..., 0])
        return Image.fromarray(f2_array)

    @mark_as_png
    def f8_仅保留遮罩里面的区域(self):
        f2_array = np.asarray(self.f2_上下扩展)
        mask = np.asarray(self.f6_转换为凸多边形)
        f2_array = np.dstack([f2_array, np.ones(f2_array.shape[:2], dtype=np.uint8) * 255]) \
            if f2_array.shape[2] == 3 else f2_array
        f2_array[..., 3] = np.where(mask == 255, f2_array[..., 3], 0)
        coords = np.column_stack(np.where(f2_array[..., 3] > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        cropped = f2_array[y_min:y_max, x_min:x_max]
        return Image.fromarray(cropped)

    @mark_as_png
    def f9_水平拉伸图像的系数_计算(self):
        array = np.asarray(self.f8_仅保留遮罩里面的区域)
        alpha = array[..., 3]
        x_min = np.where(alpha.any(axis=1), alpha.argmax(axis=1), 0)
        x_max = np.where(alpha.any(axis=1), array.shape[1] - 1 - alpha[:, ::-1].argmax(axis=1), 0)
        widths = x_max - x_min
        target = array.shape[1]
        coefficients = np.where(widths > 0, widths / target, 1.0)

        border = self.v9_不参与拉伸系数计算的边界宽度_像素
        coefficients[:border] = coefficients[border]
        coefficients[-border:] = coefficients[-border]

        smoothed = np.convolve(coefficients, np.ones(border * 2) / border / 2, mode='same')
        coefficients[border:-border] = smoothed[border:-border]

        np.save(output_path.with_suffix('.npy'), coefficients)

    @mark_as_png
    def f10_水平拉伸图像的系数_显示(self):
        coefficients = np.asarray(self.f9_水平拉伸图像的系数_计算)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(coefficients)
        ax.set_xlabel('row')
        ax.set_ylabel('coefficient')
        fig.savefig(output_path)
        plt.close(fig)

    @mark_as_png
    def f11_水平拉伸(self):
        coefficient = np.asarray(self.f9_水平拉伸图像的系数_计算)
        image = np.asarray(self.f8_仅保留遮罩里面的区域)
        border = self.v9_不参与拉伸系数计算的边界宽度_像素
        r, g, b, a = image[..., 0], image[..., 1], image[..., 2], image[..., 3]

        # 计算原始图像的坐标
        x_original, y_original = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))

        # 取透明度通道里面的第一个非空的值与最后一个非空的值作为边界，并计算每行的中心位置
        x_min = np.array([next((idx for idx, val in enumerate(row) if val > 0), 0) for row in a])
        x_max = np.array([len(row) - 1 - next((idx for idx, val in enumerate(row[::-1]) if val > 0), 0) for row in a])
        x_center = (x_min + x_max) / 2
        x_center[:border] = x_center[border]
        x_center[-border:] = x_center[-border]

        # 以每行的中心为标准进行放缩
        target_center = a.shape[1] / 2
        x_relative = x_original - target_center
        x_relative_new = x_relative * coefficient[:, np.newaxis]
        x_new = x_relative_new + x_center[:, np.newaxis]

        x_map = x_new
        y_map = y_original

        stretched_image = cv2.remap(image, x_map.astype(np.float32), y_map.astype(np.float32),
                                    interpolation=cv2.INTER_LINEAR)

        return Image.fromarray(stretched_image)

    @mark_as_png
    def f12_垂直拉伸图像的系数_计算(self):
        """计算图像垂直拉伸的系数并保存"""
        array = np.asarray(self.f11_水平拉伸)
        alpha = array[..., 3]
        y_min = np.where(alpha.any(axis=0), alpha.argmax(axis=0), 0)
        y_max = np.where(alpha.any(axis=0), array.shape[0] - 1 - alpha[:, ::-1].argmax(axis=0), 0)
        heights = y_max - y_min
        target = array.shape[0]
        coefficients = np.where(heights > 0, heights / target, 1.0)

        border = self.v12_不参与垂直拉伸系数计算的边界高度_像素
        coefficients[:border] = coefficients[border]
        coefficients[-border:] = coefficients[-border]

        smoothed = np.convolve(coefficients, np.ones(border * 2) / border / 2, mode='same')
        coefficients[border:-border] = smoothed[border:-border]

        np.save(output_path.with_suffix('.npy'), coefficients)

    @mark_as_png
    def f13_垂直拉伸图像的系数_显示(self):
        """显示垂直拉伸系数的图像"""
        coefficients = np.asarray(self.f12_垂直拉伸图像的系数_计算)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(coefficients)
        ax.set_xlabel('column')
        ax.set_ylabel('coefficient')
        fig.savefig(output_path)
        plt.close(fig)

    @mark_as_png
    def f14_垂直拉伸(self):
        """应用垂直拉伸系数进行图像拉伸"""
        coefficient = np.asarray(self.f12_垂直拉伸图像的系数_计算)
        image = np.asarray(self.f11_水平拉伸)
        border = self.v12_不参与垂直拉伸系数计算的边界高度_像素
        r, g, b, a = image[..., 0], image[..., 1], image[..., 2], image[..., 3]

        # 计算原始图像的坐标
        x_original, y_original = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))

        # 取透明度通道里面的第一个非空的值与最后一个非空的值作为边界，并计算每列的中心位置
        y_min = np.array([next((idx for idx, val in enumerate(col) if val > 0), 0) for col in
                          batch_processor.processors.base.T])
        y_max = np.array([len(col) - 1 - next((idx for idx, val in enumerate(col[::-1]) if val > 0), 0) for col in
                          batch_processor.processors.base.T])
        y_center = (y_min + y_max) / 2
        y_center[:border] = y_center[border]
        y_center[-border:] = y_center[-border]

        # 以每列的中心为标准进行放缩
        target_center = a.shape[0] / 2
        y_relative = y_original - target_center
        y_relative_new = y_relative * coefficient[np.newaxis, :]
        y_new = y_relative_new + y_center[np.newaxis, :]

        x_map = x_original
        y_map = y_new

        stretched_image = cv2.remap(image, x_map.astype(np.float32), y_map.astype(np.float32),
                                    interpolation=cv2.INTER_LINEAR)

        return Image.fromarray(stretched_image)

    @mark_as_png
    def f15_调整尺寸和裁剪(self):
        image = np.asarray(self.f14_垂直拉伸)
        border = self.v15_图像黑边
        resized = cv2.resize(image, (self.v15_最终图像宽度_像素 + border * 2, self.v15_最终图像高度_像素 + border * 2),
                             interpolation=cv2.INTER_AREA)
        cropped = resized[border:-border, border:-border]
        return Image.fromarray(cropped)

    @mark_as_png
    def f16_补全边角(self):
        image = self.get_input_image(self.f15_调整尺寸和裁剪, output_path)
        image = image.convert('RGB')
        image.save(output_path)
        raise ManuallyProcessRequiredException('使用PS补全边角')

    @mark_as_png
    def f17_根据文件名处理(self):
        image = np.asarray(self.f16_补全边角)
        match output_path.stem[-1]:
            case 'U':
                image = image[:, ::-1]
            case 'D':
                image = image.copy()
            case _:
                raise ValueError("文件名stem必须以'U'或'D'结尾")
        return Image.fromarray(image)

    @mark_as_png
    def f99_处理结果(self):
        self.get_input_image(self.f17_根据文件名处理, output_path).save(output_path)


if __name__ == '__main__':
    s25010801_花岗岩剪切前的断面光学扫描.main()
