import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


class Processor:
    base_dir = Path(r'F:\data\laser-scanner\others\侧面光学扫描的预处理')
    s1_name = r'1-原始数据'
    s2_name = r'2-转换为PNG'
    s3_name = r'3-裁剪后的PNG'
    s4_name = r'4-直方图'
    s5_name = r'5-二值化图像'
    s6_name = r'6-降噪二值化图像'
    print_lock = threading.Lock()

    def __init__(self):
        (self.base_dir / self.s2_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s3_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s4_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s5_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s6_name).mkdir(parents=True, exist_ok=True)

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
            height = image.height
            top = int(height * 0.1)
            bottom = int(height * 0.9)
            # 提取左右各100像素的边界区域，排除上下10%
            left_boundary = image.crop((0, top, 100, bottom))
            right_boundary = image.crop((image.width - 100, top, image.width, bottom))

            # 计算边界区域的平均颜色
            left_average = np.array(left_boundary).mean(axis=(0, 1))
            right_average = np.array(right_boundary).mean(axis=(0, 1))
            background_color = (left_average + right_average) / 2

            # 计算所有像素点与背景颜色的距离
            pixels = np.array(image).reshape(-1, 3)
            distances = np.linalg.norm(pixels - background_color, axis=1)

            # 使用面向对象的Matplotlib接口绘制直方图
            fig = plt.Figure()
            ax = fig.add_subplot(111)
            ax.hist(distances, bins=100, color='gray')
            ax.set_title(f'{stem} 距离直方图')
            ax.set_xlabel('距离')
            ax.set_ylabel('像素数量')
            fig.tight_layout()
            fig.savefig(output_file)
            plt.close(fig)
        self.print_safe(f"{stem} 直方图已生成并保存。")

    def s5_二值化(self, stem):
        input_file = self.base_dir / self.s3_name / f"{stem}.png"
        output_file = self.base_dir / self.s5_name / f"{stem}.png"
        if output_file.exists():
            return
        with Image.open(input_file) as image:
            height = image.height
            top = int(height * 0.1)
            bottom = int(height * 0.9)
            # 提取左右各100像素的边界区域，排除上下10%
            left_boundary = image.crop((0, top, 100, bottom))
            right_boundary = image.crop((image.width - 100, top, image.width, bottom))

            left_average = np.array(left_boundary).mean(axis=(0, 1))
            right_average = np.array(right_boundary).mean(axis=(0, 1))
            background_color = (left_average + right_average) / 2

            pixels = np.array(image).reshape(-1, 3)
            distances = np.linalg.norm(pixels - background_color, axis=1)
            threshold = np.quantile(distances, 0.7)

            binary_pixels = np.where(distances <= threshold, 0, 255).astype(np.uint8)
            binary_image = Image.fromarray(binary_pixels.reshape(image.size[1], image.size[0]), mode='L')
            binary_image.save(output_file)
        self.print_safe(f"{stem} 二值化图像已生成并保存。")

    def s6_降噪二值化(self, stem):
        input_file = self.base_dir / self.s5_name / f"{stem}.png"
        output_file = self.base_dir / self.s6_name / f"{stem}.png"
        if output_file.exists():
            return
        with Image.open(input_file) as image:
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))
            threshold = 128
            binary_pixels = np.array(blurred_image).flatten()
            binary_pixels = np.where(binary_pixels <= threshold, 0, 255).astype(np.uint8)
            denoised_image = Image.fromarray(binary_pixels.reshape(image.height, image.width), mode='L')
            denoised_image.save(output_file)
        self.print_safe(f"{stem} 降噪二值化图像已生成并保存。")

    def process_stem(self, stem):
        try:
            self.s2_将jpg格式转换为png格式(stem)
            self.s3_裁剪左右两侧(stem)
            self.s4_生成直方图(stem)
            self.s5_二值化(stem)
            self.s6_降噪二值化(stem)
        except:
            with self.print_lock:
                traceback.print_exc()

    @classmethod
    def main(cls):
        obj = cls()
        s1_dir = obj.base_dir / obj.s1_name
        stems = [file.stem for file in s1_dir.glob('*.jpg')]
        with ThreadPoolExecutor() as executor:
            executor.map(obj.process_stem, stems)


if __name__ == '__main__':
    Processor.main()
