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
    s7_name = r'7-x方向白色点数量直方图'
    s8_name = r'8-边界裁剪图像'
    s9_name = r'9-进一步边界裁剪图像'
    s10_name = r'10-纵向有效点分布直方图'
    s11_name = r'11-纵向裁剪图像'
    s12_name = r'12-进一步纵向裁剪图像'

    # 参数定义
    s3_左侧裁剪区域_像素 = 1400
    s3_右侧裁剪区域_像素 = 1000
    s4_左右边界裁剪宽度_像素 = 100
    s4_根据左右区域识别背景颜色时的上下裁剪区域_比例 = 0.1
    s5_二值化阈值_比例 = 0.7
    s6_高斯模糊半径_像素 = 10
    s8_水平裁剪过程的有效点阈值_比例 = 0.5
    s8_水平边界裁剪收缩_像素 = 10
    s10_纵向裁剪过程的有效点阈值_比例 = 0.6
    s10_纵向边界裁剪收缩_像素 = 10

    print_lock = threading.Lock()

    def __init__(self):
        for folder in [
            self.s2_name, self.s3_name, self.s4_name,
            self.s5_name, self.s6_name, self.s7_name,
            self.s8_name, self.s9_name, self.s10_name,
            self.s11_name, self.s12_name
        ]:
            (self.base_dir / folder).mkdir(parents=True, exist_ok=True)

    def print_safe(self, message):
        with self.print_lock:
            print(message)

    def s2_将jpg格式转换为png格式(self, output_path):
        input_file = self.base_dir / self.s1_name / f"{output_path.stem}.jpg"
        with Image.open(input_file) as image:
            image.save(output_path)
        self.print_safe(f"{output_path.stem} 已转换并保存。")

    def s3_裁剪左右两侧(self, output_path):
        input_file = self.base_dir / self.s2_name / f"{output_path.stem}.png"
        with Image.open(input_file) as image:
            left_crop = self.s3_左侧裁剪区域_像素
            right_crop = self.s3_右侧裁剪区域_像素
            width, height = image.size
            if width <= left_crop + right_crop:
                self.print_safe(
                    f"{output_path.stem} 图像宽度不足以截取 {left_crop} 左边和 {right_crop} 右边像素。跳过裁剪。")
                return
            cropped_image = image.crop((left_crop, 0, width - right_crop, height))
            cropped_image.save(output_path)
        self.print_safe(f"{output_path.stem} 已裁剪并保存。")

    def s4_生成直方图(self, output_path):
        input_file = self.base_dir / self.s3_name / f"{output_path.stem}.png"
        with Image.open(input_file) as image:
            height = image.height
            top = int(height * self.s4_根据左右区域识别背景颜色时的上下裁剪区域_比例)
            bottom = int(height * (1 - self.s4_根据左右区域识别背景颜色时的上下裁剪区域_比例))
            left_boundary = image.crop((0, top, self.s4_左右边界裁剪宽度_像素, bottom))
            right_boundary = image.crop((image.width - self.s4_左右边界裁剪宽度_像素, top, image.width, bottom))

            left_average = np.array(left_boundary).mean(axis=(0, 1))
            right_average = np.array(right_boundary).mean(axis=(0, 1))
            background_color = (left_average + right_average) / 2

            pixels = np.array(image).reshape(-1, 3)
            distances = np.linalg.norm(pixels - background_color, axis=1)

            fig = plt.Figure()
            ax = fig.add_subplot(111)
            ax.hist(distances, bins=100, color='gray')
            ax.set_title(f'{output_path.stem} 距离直方图')
            ax.set_xlabel('距离')
            ax.set_ylabel('像素数量')
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
        self.print_safe(f"{output_path.stem} 直方图已生成并保存。")

    def s5_二值化(self, output_path):
        input_file = self.base_dir / self.s3_name / f"{output_path.stem}.png"
        with Image.open(input_file) as image:
            height = image.height
            top = int(height * self.s4_根据左右区域识别背景颜色时的上下裁剪区域_比例)
            bottom = int(height * (1 - self.s4_根据左右区域识别背景颜色时的上下裁剪区域_比例))
            left_boundary = image.crop((0, top, self.s4_左右边界裁剪宽度_像素, bottom))
            right_boundary = image.crop((image.width - self.s4_左右边界裁剪宽度_像素, top, image.width, bottom))

            left_average = np.array(left_boundary).mean(axis=(0, 1))
            right_average = np.array(right_boundary).mean(axis=(0, 1))
            background_color = (left_average + right_average) / 2

            pixels = np.array(image).reshape(-1, 3)
            distances = np.linalg.norm(pixels - background_color, axis=1)
            threshold = np.quantile(distances, self.s5_二值化阈值_比例)

            binary_pixels = np.where(distances <= threshold, 0, 255).astype(np.uint8)
            binary_image = Image.fromarray(binary_pixels.reshape(image.size[1], image.size[0]), mode='L')
            binary_image.save(output_path)
        self.print_safe(f"{output_path.stem} 二值化图像已生成并保存。")

    def s6_降噪二值化(self, output_path):
        input_file = self.base_dir / self.s5_name / f"{output_path.stem}.png"
        with Image.open(input_file) as image:
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=self.s6_高斯模糊半径_像素))
            threshold = 128
            binary_pixels = np.array(blurred_image).flatten()
            binary_pixels = np.where(binary_pixels <= threshold, 0, 255).astype(np.uint8)
            denoised_image = Image.fromarray(binary_pixels.reshape(image.height, image.width), mode='L')
            denoised_image.save(output_path)
        self.print_safe(f"{output_path.stem} 降噪二值化图像已生成并保存。")

    def s7_绘制x方向白色点数量直方图(self, output_path):
        input_file = self.base_dir / self.s5_name / f"{output_path.stem}.png"
        with Image.open(input_file) as image:
            binary_array = np.array(image)
            white_counts = np.sum(binary_array > 128, axis=0)

            fig = plt.Figure()
            ax = fig.add_subplot(111)
            ax.plot(range(len(white_counts)), white_counts, color='blue')
            ax.set_title(f'{output_path.stem} x方向白色点数量')
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('白色点数量')
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
        self.print_safe(f"{output_path.stem} x方向白色点数量图已生成并保存。")

    def s8_边界裁剪图像(self, output_path):
        input_file = self.base_dir / self.s5_name / f"{output_path.stem}.png"
        with Image.open(input_file) as image:
            binary_array = np.array(image)
            height, width = binary_array.shape

            white_counts = np.sum(binary_array > 128, axis=0)

            max_count = white_counts.max()
            threshold = self.s8_水平裁剪过程的有效点阈值_比例 * max_count

            left_boundary_candidates = np.where(white_counts > threshold)[0]
            if left_boundary_candidates.size == 0:
                self.print_safe(f"{output_path.stem} 没有检测到满足阈值的白色点，跳过边界裁剪。")
                return
            left_boundary = left_boundary_candidates.min()

            right_boundary_candidates = np.where(white_counts > threshold)[0]
            right_boundary = right_boundary_candidates.max()

            left_boundary = min(left_boundary + self.s8_水平边界裁剪收缩_像素, width)
            right_boundary = max(right_boundary - self.s8_水平边界裁剪收缩_像素, 0)

            if left_boundary >= right_boundary:
                self.print_safe(f"{output_path.stem} 边界裁剪后区域无效，跳过裁剪。")
                return

            cropped_image = image.crop((left_boundary, 0, right_boundary, height))
            cropped_image.save(output_path)
        self.print_safe(f"{output_path.stem} 边界裁剪图像已生成并保存。")

    def s9_进一步边界裁剪图像(self, output_path):
        binary_input_file = self.base_dir / self.s5_name / f"{output_path.stem}.png"
        color_input_file = self.base_dir / self.s3_name / f"{output_path.stem}.png"

        with Image.open(binary_input_file) as binary_image:
            if binary_image.mode != 'L':
                binary_image = binary_image.convert("L")
            binary_array = np.array(binary_image)

            if binary_array.ndim != 2:
                self.print_safe(f"{output_path.stem} 二值化图像不是二维的，无法进行边界裁剪。")
                return

            height, width = binary_array.shape

            white_counts = np.sum(binary_array > 128, axis=0)

            max_count = white_counts.max()
            threshold = self.s8_水平裁剪过程的有效点阈值_比例 * max_count

            left_boundary_candidates = np.where(white_counts > threshold)[0]
            if left_boundary_candidates.size == 0:
                self.print_safe(f"{output_path.stem} 没有检测到满足阈值的白色点，跳过进一步边界裁剪。")
                return
            left_boundary = left_boundary_candidates.min()

            right_boundary_candidates = np.where(white_counts > threshold)[0]
            right_boundary = right_boundary_candidates.max()

            left_boundary = min(left_boundary + self.s8_水平边界裁剪收缩_像素, width)
            right_boundary = max(right_boundary - self.s8_水平边界裁剪收缩_像素, 0)

            if left_boundary >= right_boundary:
                self.print_safe(f"{output_path.stem} 进一步边界裁剪后区域无效，跳过裁剪。")
                return

        with Image.open(color_input_file) as color_image:
            cropped_image = color_image.crop((left_boundary, 0, right_boundary, height))
            cropped_image.save(output_path)
        self.print_safe(f"{output_path.stem} 进一步边界裁剪图像已生成并保存。")

    def s10_生成纵向有效点分布直方图(self, output_path):
        input_file = self.base_dir / self.s8_name / f"{output_path.stem}.png"
        with Image.open(input_file) as image:
            binary_array = np.array(image)
            white_counts = np.sum(binary_array > 128, axis=1)

            fig = plt.Figure()
            ax = fig.add_subplot(111)
            ax.plot(range(len(white_counts)), white_counts, color='green')
            ax.set_title(f'{output_path.stem} 纵向有效点分布')
            ax.set_xlabel('Y 坐标')
            ax.set_ylabel('有效点数量')
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
        self.print_safe(f"{output_path.stem} 纵向有效点分布图已生成并保存。")

    def s11_纵向裁剪图像(self, output_path):
        input_file = self.base_dir / self.s8_name / f"{output_path.stem}.png"
        with Image.open(input_file) as image:
            binary_array = np.array(image)
            height, width = binary_array.shape

            white_counts = np.sum(binary_array > 128, axis=1)

            max_count = white_counts.max()
            threshold = self.s10_纵向裁剪过程的有效点阈值_比例 * max_count

            top_boundary_candidates = np.where(white_counts > threshold)[0]
            if top_boundary_candidates.size == 0:
                self.print_safe(f"{output_path.stem} 没有检测到满足阈值的白色点，跳过纵向裁剪。")
                return
            top_boundary = top_boundary_candidates.min()

            bottom_boundary_candidates = np.where(white_counts > threshold)[0]
            bottom_boundary = bottom_boundary_candidates.max()

            top_boundary = min(top_boundary + self.s10_纵向边界裁剪收缩_像素, height)
            bottom_boundary = max(bottom_boundary - self.s10_纵向边界裁剪收缩_像素, 0)

            if top_boundary >= bottom_boundary:
                self.print_safe(f"{output_path.stem} 纵向裁剪后区域无效，跳过裁剪。")
                return

            cropped_image = image.crop((0, top_boundary, width, bottom_boundary))
            cropped_image.save(output_path)
        self.print_safe(f"{output_path.stem} 纵向裁剪图像已生成并保存。")

    def s12_进一步纵向裁剪图像(self, output_path):
        binary_input_file = self.base_dir / self.s8_name / f"{output_path.stem}.png"
        color_input_file = self.base_dir / self.s9_name / f"{output_path.stem}.png"

        with Image.open(binary_input_file) as binary_image:
            if binary_image.mode != 'L':
                binary_image = binary_image.convert("L")
            binary_array = np.array(binary_image)

            if binary_array.ndim != 2:
                self.print_safe(f"{output_path.stem} 二值化图像不是二维的，无法进行纵向裁剪。")
                return

            height, width = binary_array.shape

            white_counts = np.sum(binary_array > 128, axis=1)

            max_count = white_counts.max()
            threshold = self.s10_纵向裁剪过程的有效点阈值_比例 * max_count

            top_boundary_candidates = np.where(white_counts > threshold)[0]
            if top_boundary_candidates.size == 0:
                self.print_safe(f"{output_path.stem} 没有检测到满足阈值的白色点，跳过进一步纵向裁剪。")
                return
            top_boundary = top_boundary_candidates.min()

            bottom_boundary_candidates = np.where(white_counts > threshold)[0]
            bottom_boundary = bottom_boundary_candidates.max()

            top_boundary = min(top_boundary + self.s10_纵向边界裁剪收缩_像素, height)
            bottom_boundary = max(bottom_boundary - self.s10_纵向边界裁剪收缩_像素, 0)

            if top_boundary >= bottom_boundary:
                self.print_safe(f"{output_path.stem} 进一步纵向裁剪后区域无效，跳过裁剪。")
                return

        with Image.open(color_input_file) as color_image:
            cropped_image = color_image.crop((0, top_boundary, width, bottom_boundary))
            cropped_image.save(output_path)
        self.print_safe(f"{output_path.stem} 进一步纵向裁剪图像已生成并保存。")

    def process_stem(self, stem):
        try:
            steps = [
                (self.s2_name, self.s2_将jpg格式转换为png格式),
                (self.s3_name, self.s3_裁剪左右两侧),
                (self.s4_name, self.s4_生成直方图),
                (self.s5_name, self.s5_二值化),
                # (self.s6_name, self.s6_降噪二值化),
                (self.s7_name, self.s7_绘制x方向白色点数量直方图),
                (self.s8_name, self.s8_边界裁剪图像),
                (self.s9_name, self.s9_进一步边界裁剪图像),
                (self.s10_name, self.s10_生成纵向有效点分布直方图),
                (self.s11_name, self.s11_纵向裁剪图像),
                (self.s12_name, self.s12_进一步纵向裁剪图像),
            ]
            for folder, func in steps:
                output_path = self.base_dir / folder / f"{stem}.png"
                if output_path.exists():
                    continue
                func(output_path)
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
