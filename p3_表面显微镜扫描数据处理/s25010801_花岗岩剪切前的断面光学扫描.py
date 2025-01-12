from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from p3_表面显微镜扫描数据处理.base import BaseProcessor, ManuallyProcessRequiredException


class Processor(BaseProcessor):
    def __init__(self):
        self.base_dir = Path(r'F:\data\laser-scanner\25010801-花岗岩剪切前的断面光学扫描')
        self.source_file_function = self.f1_原始数据
        self.final_file_function = self.f99_处理结果
        self.step_functions = [
            self.f1_原始数据,
            self.f2_上下扩展,
            self.f3_删除非主体的像素,
            self.f4_非透明部分的mask,
            self.f5_转换为凸边界,
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
        """将非透明部分设置为白色，透明部分设置为黑色"""
        array = self.get_input_array(self.f3_删除非主体的像素, output_path)
        mask = (array[..., 3] > 0).astype(np.uint8) * 255
        Image.fromarray(mask, 'L').save(output_path)

    def f5_转换为凸边界(self, output_path: Path):
        """识别最大区域并将其边界转换为凸多边形，最后保存为黑白图像"""
        array = self.get_input_array(self.f4_非透明部分的mask, output_path)
        contours = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        largest = max(contours, key=cv2.contourArea)
        convex = np.zeros_like(array)
        cv2.drawContours(convex, [largest], -1, (255,), thickness=cv2.FILLED)
        Image.fromarray(convex, 'L').save(output_path)

    def f99_处理结果(self, output_path: Path):
        raise ManuallyProcessRequiredException


if __name__ == '__main__':
    Processor.main()
