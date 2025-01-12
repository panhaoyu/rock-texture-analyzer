from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from more_itertools import only

from p3_表面显微镜扫描数据处理.base import BaseProcessor, ManuallyProcessRequiredException


class Processor(BaseProcessor):
    v6_转换为凸多边形的检测长度_像素 = 100
    v9_目标长度_像素 = 4000
    v9_目标宽度_像素 = 4000

    def __init__(self):
        self.base_dir = Path(r'F:\data\laser-scanner\25010801-花岗岩剪切前的断面光学扫描')
        self.source_file_function = self.f1_原始数据
        self.final_file_function = self.f99_处理结果
        self.step_functions = [
            self.f1_原始数据,
            self.f2_上下扩展,
            self.f3_删除非主体的像素,
            self.f4_非透明部分的mask,
            self.f5_提取最大的区域,
            self.f6_转换为凸多边形,
            self.f7_显示识别效果,
            self.f8_仅保留遮罩里面的区域,
            self.f9_水平拉伸图像的系数_计算,
            self.f10_水平拉伸图像的系数_显示,
            self.f99_处理结果,
        ]

    def f1_原始数据(self, output_path: Path):
        raise ManuallyProcessRequiredException

    def f2_上下扩展(self, output_path: Path):
        array = self.get_input_array(self.f1_原始数据, output_path)
        extended_array = np.pad(array, ((1000, 1000), (0, 0), (0, 0)), mode='edge')
        Image.fromarray(extended_array).save(output_path)

    def f3_删除非主体的像素(self, output_path: Path):
        raise ManuallyProcessRequiredException

    def f4_非透明部分的mask(self, output_path: Path):
        array = self.get_input_array(self.f3_删除非主体的像素, output_path)
        mask = (array[..., 3] > 0).astype(np.uint8) * 255
        Image.fromarray(mask, 'L').save(output_path)

    def f5_提取最大的区域(self, output_path: Path):
        array = self.get_input_array(self.f4_非透明部分的mask, output_path)
        contours = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        largest = max(contours, key=cv2.contourArea)
        convex = np.zeros_like(array)
        cv2.drawContours(convex, [largest], -1, (255,), thickness=cv2.FILLED)
        Image.fromarray(convex, 'L').save(output_path)

    def f6_转换为凸多边形(self, output_path: Path):
        array = self.get_input_array(self.f5_提取最大的区域, output_path)
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
        Image.fromarray(convex, 'L').save(output_path)

    def f7_显示识别效果(self, output_path: Path):
        f2_array = self.get_input_array(self.f2_上下扩展, output_path)
        f6_array = self.get_input_array(self.f6_转换为凸多边形, output_path)
        f2_array[..., 0] = np.where(f6_array == 255, 255, f2_array[..., 0])
        Image.fromarray(f2_array).save(output_path)

    def f8_仅保留遮罩里面的区域(self, output_path: Path):
        f2_array = self.get_input_array(self.f2_上下扩展, output_path)
        mask = self.get_input_array(self.f6_转换为凸多边形, output_path)
        f2_array = np.dstack([f2_array, np.ones(f2_array.shape[:2], dtype=np.uint8) * 255]) \
            if f2_array.shape[2] == 3 else f2_array
        f2_array[..., 3] = np.where(mask == 255, f2_array[..., 3], 0)
        coords = np.column_stack(np.where(f2_array[..., 3] > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        cropped = f2_array[y_min:y_max, x_min:x_max]
        Image.fromarray(cropped).save(output_path)

    def f9_水平拉伸图像的系数_计算(self, output_path: Path):
        array = self.get_input_array(self.f8_仅保留遮罩里面的区域, output_path)
        alpha = array[..., 3]
        x_min = np.where(alpha.any(axis=1), alpha.argmax(axis=1), 0)
        x_max = np.where(alpha.any(axis=1), array.shape[1] - 1 - alpha[:, ::-1].argmax(axis=1), 0)
        widths = x_max - x_min
        coefficients = np.where(widths > 0, widths / self.v9_目标长度_像素, 1.0)

        border = 100
        coefficients[:border] = coefficients[border]
        coefficients[-border:] = coefficients[-border]

        smoothed = np.convolve(coefficients, np.ones(border * 2) / border / 2, mode='same')
        coefficients[border:-border] = smoothed[border:-border]

        np.save(output_path.with_suffix('.npy'), coefficients)

    def f10_水平拉伸图像的系数_显示(self, output_path: Path):
        coefficients = np.load(self.get_input_path(self.f9_水平拉伸图像的系数_计算, output_path))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(coefficients)
        ax.set_xlabel('row')
        ax.set_ylabel('coefficient')
        fig.savefig(output_path)
        plt.close(fig)

    def f99_处理结果(self, output_path: Path):
        raise ManuallyProcessRequiredException


if __name__ == '__main__':
    Processor.main()
