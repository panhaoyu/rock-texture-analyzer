import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Processor:
    base_dir = Path(r'F:\data\laser-scanner\others\侧面光学扫描的预处理')
    s1_name = r'1-原始数据'
    s2_name = r'2-转换为PNG'
    s3_name = r'3-裁剪后的PNG'
    s4_name = r'4-直方图'
    print_lock = threading.Lock()

    def __init__(self):
        (self.base_dir / self.s2_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s3_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s4_name).mkdir(parents=True, exist_ok=True)

    def print_safe(self, message):
        with self.print_lock:
            print(message)

    def s2_将jpg格式转换为png格式(self, stem):
        input_file = self.base_dir / self.s1_name / f"{stem}.jpg"
        output_file = self.base_dir / self.s2_name / f"{stem}.png"
        if output_file.exists():
            return
        with Image.open(input_file) as image:
            image.save(output_file)
        self.print_safe(f"{stem} 已转换并保存。")

    def s3_裁剪左右两侧(self, stem):
        input_file = self.base_dir / self.s2_name / f"{stem}.png"
        output_file = self.base_dir / self.s3_name / f"{stem}.png"
        if output_file.exists():
            return
        with Image.open(input_file) as image:
            left_crop = 1400
            right_crop = 1000
            width, height = image.size
            if width <= left_crop + right_crop:
                self.print_safe(f"{stem} 图像宽度不足以截取 {left_crop} 左边和 {right_crop} 右边像素。跳过裁剪。")
                return
            cropped_image = image.crop((left_crop, 0, width - right_crop, height))
            cropped_image.save(output_file)
        self.print_safe(f"{stem} 已裁剪并保存。")

    def s4_生成直方图(self, stem):
        input_file = self.base_dir / self.s3_name / f"{stem}.png"
        output_file = self.base_dir / self.s4_name / f"{stem}.png"
        if output_file.exists():
            return
        with Image.open(input_file) as image:
            # 提取左右各100像素的边界区域
            left_boundary = image.crop((0, 0, 100, image.height))
            right_boundary = image.crop((image.width - 100, 0, image.width, image.height))

            # 计算边界区域的平均颜色
            left_average = np.array(left_boundary).mean()
            right_average = np.array(right_boundary).mean()
            background_color = (left_average + right_average) / 2

            # 计算所有像素点与背景颜色的距离
            pixels = np.array(image).flatten()
            distances = np.abs(pixels - background_color)

            # 绘制距离直方图
            plt.figure()
            plt.hist(distances, bins=100)
            plt.title(f'{stem} 距离直方图')
            plt.xlabel('距离')
            plt.ylabel('像素数量')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
        self.print_safe(f"{stem} 直方图已生成并保存。")

    def process_stem(self, stem):
        self.s2_将jpg格式转换为png格式(stem)
        self.s3_裁剪左右两侧(stem)
        self.s4_生成直方图(stem)

    @classmethod
    def main(cls):
        obj = cls()
        s1_dir = obj.base_dir / obj.s1_name
        stems = [file.stem for file in s1_dir.glob('*.jpg')]
        with ThreadPoolExecutor() as executor:
            executor.map(obj.process_stem, stems)


if __name__ == '__main__':
    Processor.main()
