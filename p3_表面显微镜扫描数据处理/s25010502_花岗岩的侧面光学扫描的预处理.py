from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, ImageFilter

from rock_texture_analyzer.base import BaseProcessor, ManuallyProcessRequiredException, mark_as_method
from p3_表面显微镜扫描数据处理.utils.s1_图像补全_阿里云 import erase_image_with_oss


class s25010502_花岗岩的侧面光学扫描的预处理(BaseProcessor):
    p3_左侧裁剪区域_像素: int = 1400
    p3_右侧裁剪区域_像素: int = 1000
    p4_左右边界裁剪宽度_像素: int = 100
    p4_根据左右区域识别背景颜色时的上下裁剪区域_比例: float = 0.1
    # p5_二值化阈值_比例: float = 0.8  # 花岗岩
    p5_二值化阈值_比例: float = 0.9  # 砾岩
    p6_高斯模糊半径_像素: int = 10
    p8_水平裁剪过程的有效点阈值_比例: float = 0.5
    p8_水平边界裁剪收缩_像素: int = 10
    p10_纵向裁剪过程的有效点阈值_比例: float = 0.6
    p10_纵向边界裁剪收缩_像素: int = 10
    # p14_亮度最小值: float = 10  # 花岗岩
    # p14_亮度最大值: float = 125  # 花岗岩
    p14_亮度最小值: float = 20  # 砾岩
    p14_亮度最大值: float = 120  # 砾岩
    p16_直方图平滑窗口半径_像素: int = 10
    p17_缩放图像大小: tuple[int, int] = (4000, 4000)
    # p18_补全时的上下裁剪范围_像素: int = 1200  # 花岗岩
    p18_补全时的上下裁剪范围_像素: int = 1300  # 砾岩
    v19_识别黑线时的范围扩大像素数量: float = 10
    v19_识别黑线时的阈值扩大系数: float = 1.5
    v20_识别黑线时的掩膜膨胀半径: int = 5

    @mark_as_method
    def f1_原始数据(self, output_path: Path) -> None:
        raise ManuallyProcessRequiredException

    @mark_as_method
    def f2_将jpg格式转换为png格式(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f1_原始数据, output_path.stem)
        with Image.open(input_path) as image:
            image.save(output_path)

    @mark_as_method
    def f3_裁剪左右两侧(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f2_将jpg格式转换为png格式, output_path.stem)
        with Image.open(input_path) as image:
            left_crop: int = self.p3_左侧裁剪区域_像素
            right_crop: int = self.p3_右侧裁剪区域_像素
            width, height = image.size
            if width <= left_crop + right_crop:
                self.print_safe(
                    f"{output_path.stem} 图像宽度不足以截取 {left_crop} 左边和 {right_crop} 右边像素。跳过裁剪。")
                return
            cropped_image: Image.Image = image.crop((left_crop, 0, width - right_crop, height))
            cropped_image.save(output_path)

    @mark_as_method
    def f4_生成直方图(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f3_裁剪左右两侧, output_path.stem)
        with Image.open(input_path) as image:
            height: int = image.height
            top: int = int(height * self.p4_根据左右区域识别背景颜色时的上下裁剪区域_比例)
            bottom: int = int(height * (1 - self.p4_根据左右区域识别背景颜色时的上下裁剪区域_比例))
            left_boundary: Image.Image = image.crop((0, top, self.p4_左右边界裁剪宽度_像素, bottom))
            right_boundary: Image.Image = image.crop(
                (image.width - self.p4_左右边界裁剪宽度_像素, top, image.width, bottom))
            left_average: np.ndarray = np.array(left_boundary).mean(axis=(0, 1))
            right_average: np.ndarray = np.array(right_boundary).mean(axis=(0, 1))
            background_color: np.ndarray = (left_average + right_average) / 2
            pixels: np.ndarray = np.array(image).reshape(-1, 3)
            distances: np.ndarray = np.linalg.norm(pixels - background_color, axis=1)
        fig: plt.Figure = plt.Figure()
        ax: plt.Axes = fig.add_subplot(111)
        ax.hist(distances, bins=100, color='gray')
        ax.set_title(f'{output_path.stem} distance histogram')
        ax.set_xlabel('distance')
        ax.set_ylabel('number of pixels')
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    @mark_as_method
    def f5_二值化(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f3_裁剪左右两侧, output_path.stem)
        with Image.open(input_path) as image:
            height: int = image.height
            top: int = int(height * self.p4_根据左右区域识别背景颜色时的上下裁剪区域_比例)
            bottom: int = int(height * (1 - self.p4_根据左右区域识别背景颜色时的上下裁剪区域_比例))
            left_boundary: Image.Image = image.crop((0, top, self.p4_左右边界裁剪宽度_像素, bottom))
            right_boundary: Image.Image = image.crop(
                (image.width - self.p4_左右边界裁剪宽度_像素, top, image.width, bottom))
            left_average: np.ndarray = np.array(left_boundary).mean(axis=(0, 1))
            right_average: np.ndarray = np.array(right_boundary).mean(axis=(0, 1))
            background_color: np.ndarray = (left_average + right_average) / 2
            pixels: np.ndarray = np.array(image).reshape(-1, 3)
            distances: np.ndarray = np.linalg.norm(pixels - background_color, axis=1)
            threshold: float = np.quantile(distances, np.double(self.p5_二值化阈值_比例))
            binary_pixels: np.ndarray = np.where(distances <= threshold, 0, 255).astype(np.uint8)
            binary_image: Image.Image = Image.fromarray(binary_pixels.reshape(image.size[1], image.size[0]), mode='L')
            binary_image.save(output_path)

    @mark_as_method
    def f6_降噪二值化(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f5_二值化, output_path.stem)
        with Image.open(input_path) as image:
            blurred_image: Image.Image = image.filter(ImageFilter.GaussianBlur(radius=self.p6_高斯模糊半径_像素))
            threshold: int = 128
            binary_pixels: np.ndarray = np.array(blurred_image).flatten()
            binary_pixels = np.where(binary_pixels <= threshold, 0, 255).astype(np.uint8)
            denoised_image: Image.Image = Image.fromarray(binary_pixels.reshape(image.height, image.width), mode='L')
            denoised_image.save(output_path)

    @mark_as_method
    def f7_绘制x方向白色点数量直方图(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f5_二值化, output_path.stem)
        with Image.open(input_path) as image:
            binary_array: np.ndarray = np.array(image)
            white_counts: np.ndarray = np.sum(binary_array > 128, axis=0)
        fig: plt.Figure = plt.Figure()
        ax: plt.Axes = fig.add_subplot(111)
        ax.plot(range(len(white_counts)), white_counts, color='blue')
        ax.set_title(f'{output_path.stem} x-direction white pixel count')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('number of white pixels')
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    @mark_as_method
    def f8_边界裁剪图像(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f5_二值化, output_path.stem)
        with Image.open(input_path) as image:
            binary_array: np.ndarray = np.array(image)
            height, width = binary_array.shape
            white_counts: np.ndarray = np.sum(binary_array > 128, axis=0)
            max_count: int = white_counts.max()
            threshold: float = self.p8_水平裁剪过程的有效点阈值_比例 * max_count
            left_candidates: np.ndarray = np.where(white_counts > threshold)[0]
            if left_candidates.size == 0:
                self.print_safe(f"{output_path.stem} 没有检测到满足阈值的白色点，跳过边界裁剪。")
                return
            left_boundary: int = left_candidates.min()
            right_candidates: np.ndarray = np.where(white_counts > threshold)[0]
            right_boundary: int = right_candidates.max()
            left_boundary = min(left_boundary + self.p8_水平边界裁剪收缩_像素, width)
            right_boundary = max(right_boundary - self.p8_水平边界裁剪收缩_像素, 0)
            if left_boundary >= right_boundary:
                self.print_safe(f"{output_path.stem} 边界裁剪后区域无效，跳过裁剪。")
                return
            cropped_image: Image.Image = image.crop((left_boundary, 0, right_boundary, height))
            cropped_image.save(output_path)

    @mark_as_method
    def f9_进一步边界裁剪图像(self, output_path: Path) -> None:
        binary_input_path: Path = self.get_file_path(self.f5_二值化, output_path.stem)
        color_input_path: Path = self.get_file_path(self.f3_裁剪左右两侧, output_path.stem)
        with Image.open(binary_input_path) as binary_image:
            if binary_image.mode != 'L':
                binary_image = binary_image.convert("L")
            binary_array: np.ndarray = np.array(binary_image)
            if binary_array.ndim != 2:
                self.print_safe(f"{output_path.stem} 二值化图像不是二维的，无法进行边界裁剪。")
                return
            height, width = binary_array.shape
            white_counts: np.ndarray = np.sum(binary_array > 128, axis=0)
            max_count: int = white_counts.max()
            threshold: float = self.p8_水平裁剪过程的有效点阈值_比例 * max_count
            left_candidates: np.ndarray = np.where(white_counts > threshold)[0]
            if left_candidates.size == 0:
                self.print_safe(f"{output_path.stem} 没有检测到满足阈值的白色点，跳过进一步边界裁剪。")
                return
            left_boundary: int = left_candidates.min()
            right_candidates: np.ndarray = np.where(white_counts > threshold)[0]
            right_boundary: int = right_candidates.max()
            left_boundary = min(left_boundary + self.p8_水平边界裁剪收缩_像素, width)
            right_boundary = max(right_boundary - self.p8_水平边界裁剪收缩_像素, 0)
            if left_boundary >= right_boundary:
                self.print_safe(f"{output_path.stem} 进一步边界裁剪后区域无效，跳过裁剪。")
                return
        with Image.open(color_input_path) as color_image:
            cropped_image: Image.Image = color_image.crop((left_boundary, 0, right_boundary, height))
            cropped_image.save(output_path)

    @mark_as_method
    def f10_生成纵向有效点分布直方图(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f8_边界裁剪图像, output_path.stem)
        with Image.open(input_path) as image:
            binary_array: np.ndarray = np.array(image)
            white_counts: np.ndarray = np.sum(binary_array > 128, axis=1)
        fig: plt.Figure = plt.Figure()
        ax: plt.Axes = fig.add_subplot(111)
        ax.plot(range(len(white_counts)), white_counts, color='green')
        ax.set_title(f'{output_path.stem} vertical distribution of effective points')
        ax.set_xlabel('Y coordinate')
        ax.set_ylabel('number of effective points')
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    @mark_as_method
    def f11_纵向裁剪图像(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f8_边界裁剪图像, output_path.stem)
        with Image.open(input_path) as image:
            binary_array: np.ndarray = np.array(image)
            height, width = binary_array.shape
            white_counts: np.ndarray = np.sum(binary_array > 128, axis=1)
            max_count: int = white_counts.max()
            threshold: float = self.p10_纵向裁剪过程的有效点阈值_比例 * max_count
            top_candidates: np.ndarray = np.where(white_counts > threshold)[0]
            if top_candidates.size == 0:
                self.print_safe(f"{output_path.stem} 没有检测到满足阈值的白色点，跳过纵向裁剪。")
                return
            top_boundary: int = top_candidates.min()
            bottom_candidates: np.ndarray = np.where(white_counts > threshold)[0]
            bottom_boundary: int = bottom_candidates.max()
            top_boundary = min(top_boundary + self.p10_纵向边界裁剪收缩_像素, height)
            bottom_boundary = max(bottom_boundary - self.p10_纵向边界裁剪收缩_像素, 0)
            if top_boundary >= bottom_boundary:
                self.print_safe(f"{output_path.stem} 纵向裁剪后区域无效，跳过裁剪。")
                return
            cropped_image: Image.Image = image.crop((0, top_boundary, width, bottom_boundary))
            cropped_image.save(output_path)

    @mark_as_method
    def f12_进一步纵向裁剪图像(self, output_path: Path) -> None:
        binary_input_path: Path = self.get_file_path(self.f8_边界裁剪图像, output_path.stem)
        color_input_path: Path = self.get_file_path(self.f9_进一步边界裁剪图像, output_path.stem)
        with Image.open(binary_input_path) as binary_image:
            if binary_image.mode != 'L':
                binary_image = binary_image.convert("L")
            binary_array: np.ndarray = np.array(binary_image)
            if binary_array.ndim != 2:
                self.print_safe(f"{output_path.stem} 二值化图像不是二维的，无法进行纵向裁剪。")
                return
            height, width = binary_array.shape
            white_counts: np.ndarray = np.sum(binary_array > 128, axis=1)
            max_count: int = white_counts.max()
            threshold: float = self.p10_纵向裁剪过程的有效点阈值_比例 * max_count
            top_candidates: np.ndarray = np.where(white_counts > threshold)[0]
            if top_candidates.size == 0:
                self.print_safe(f"{output_path.stem} 没有检测到满足阈值的白色点，跳过进一步纵向裁剪。")
                return
            top_boundary: int = top_candidates.min()
            bottom_candidates: np.ndarray = np.where(white_counts > threshold)[0]
            bottom_boundary: int = bottom_candidates.max()
            top_boundary = min(top_boundary + self.p10_纵向边界裁剪收缩_像素, height)
            bottom_boundary = max(bottom_boundary - self.p10_纵向边界裁剪收缩_像素, 0)
            if top_boundary >= bottom_boundary:
                self.print_safe(f"{output_path.stem} 进一步纵向裁剪后区域无效，跳过裁剪。")
                return
        with Image.open(color_input_path) as color_image:
            cropped_image: Image.Image = color_image.crop((0, top_boundary, width, bottom_boundary))
            cropped_image.save(output_path)

    @mark_as_method
    def f13_亮度直方图(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f12_进一步纵向裁剪图像, output_path.stem)
        with Image.open(input_path) as image:
            assert image.mode == 'RGB'
            lab: Image.Image = image.convert('LAB')
            l, _, _ = lab.split()
            l_array: np.ndarray = np.asarray(l).ravel()
        self.print_safe(f'{np.quantile(l_array, 0.05)=} {np.quantile(l_array, 0.95)=}')
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot(111)
        ax.hist(l_array, bins=256)
        ax.set_xlim(0, 160)
        fig.savefig(output_path)
        plt.close(fig)

    @mark_as_method
    def f14_调整亮度(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f12_进一步纵向裁剪图像, output_path.stem)
        with Image.open(input_path) as image:
            lab: Image.Image = image.convert('LAB')
            l, a, b = lab.split()
            l_array: np.ndarray = np.asarray(l).astype(np.float32)
            l_scaled = (l_array - self.p14_亮度最小值) / (self.p14_亮度最大值 - self.p14_亮度最小值) * 255
            l_scaled: np.ndarray = np.clip(l_scaled, a_min=0, a_max=255).astype(np.uint8)
            l = Image.fromarray(l_scaled, mode='L')
            lab = Image.merge('LAB', (l, a, b))
            rgb: Image.Image = lab.convert('RGB')
            rgb.save(output_path)

    @mark_as_method
    def f16_绘制RGB的KDE(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f14_调整亮度, output_path.stem)
        with Image.open(input_path) as image:
            rgb_array: np.ndarray = np.array(image)
        fig: plt.Figure = plt.Figure()
        ax: plt.Axes = fig.add_subplot(111)
        colors = ['red', 'green', 'blue']
        size = self.p16_直方图平滑窗口半径_像素 * 2 - 1
        kernel = np.ones(size) / size
        for channel, color in zip(rgb_array.reshape(-1, 3).T, colors):
            counts, bin_edges = np.histogram(channel, bins=256, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            smoothed_counts = np.convolve(counts, kernel, mode='same')
            ax.plot(bin_centers, smoothed_counts, color=color)
        ax.set_title(f'{output_path.stem} RGB KDE')
        ax.set_xlabel('pixel value')
        ax.set_ylabel('density')
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    @mark_as_method
    def f17_缩放图像(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f14_调整亮度, output_path.stem)
        with Image.open(input_path) as image:
            resized_image: Image.Image = image.resize(self.p17_缩放图像大小)
            resized_image.save(output_path)

    @mark_as_method
    def f18_需要补全的区域(self, output_path: Path) -> None:
        input_path = self.get_file_path(self.f17_缩放图像, output_path.stem)
        with Image.open(input_path) as image:
            image = np.asarray(image)[self.p18_补全时的上下裁剪范围_像素:-self.p18_补全时的上下裁剪范围_像素, :, :]
            Image.fromarray(image).save(output_path)

    @mark_as_method
    def f19_识别黑色水平线区域(self, output_path: Path) -> None:
        input_path = self.get_file_path(self.f18_需要补全的区域, output_path.stem)
        with Image.open(input_path) as image:
            gray = image.convert('L')
            pixels = np.array(gray)
            threshold = np.percentile(pixels, 40)
            binary = pixels <= threshold
            black_counts = binary.sum(axis=1)
            mid_y = pixels.shape[0] // 2
            high_black = black_counts > (0.5 * pixels.shape[1])

            indices = np.where(high_black)[0]
            expand = self.v19_识别黑线时的范围扩大像素数量
            closest_idx = int(indices[np.argmin(np.abs(indices - mid_y))])
            start = max(closest_idx, 0)
            end = min(closest_idx, len(high_black) - 1)
            while start > 0 and high_black[start - 1]:
                start = max(start - 1, 0)
            while end < len(high_black) - 1 and high_black[end + 1]:
                end = min(end + 1, len(high_black) - 1)
            start -= expand
            end += expand
            mask = np.zeros_like(pixels, dtype=np.uint8)
            mask[start:end + 1, :] = 255
            center_pixels = pixels[start:end + 1, :]
            center_pixels = np.where(center_pixels < threshold * self.v19_识别黑线时的阈值扩大系数, 255, 0)
            mask[start:end + 1, :] = center_pixels
        mask_image = Image.fromarray(mask, mode='L')
        mask_image.save(output_path)

    @mark_as_method
    def f20_膨胀白色部分(self, output_path: Path) -> None:
        input_path = self.get_file_path(self.f19_识别黑色水平线区域, output_path.stem)
        size = self.v20_识别黑线时的掩膜膨胀半径
        with Image.open(input_path) as image:
            dilated_image = image.filter(ImageFilter.MaxFilter(size * 2 - 1))
            dilated_image.save(output_path)

    @mark_as_method
    def f21_翻转黑白区域(self, output_path: Path) -> None:
        input_path = self.get_file_path(self.f20_膨胀白色部分, output_path.stem)
        with Image.open(input_path) as image:
            binary = np.array(image)
            inverted = np.where(binary == 255, 0, 255).astype(np.uint8)
            Image.fromarray(inverted, mode='L').save(output_path)

    @mark_as_method
    def f22_补全黑线(self, output_path: Path) -> None:
        base_dir = self.base_dir
        local_image_path = self.get_file_path(self.f18_需要补全的区域, output_path.stem)
        local_mask_path = self.get_file_path(self.f20_膨胀白色部分, output_path.stem)
        local_foreground_path = self.get_file_path(self.f21_翻转黑白区域, output_path.stem)

        url: str = erase_image_with_oss(base_dir, local_image_path, local_mask_path, local_foreground_path)
        response = requests.get(url)
        response.raise_for_status()

        foreground_image = Image.open(BytesIO(response.content))
        foreground_image.save(output_path)

    @mark_as_method
    def f23_合并补全图像(self, output_path: Path) -> None:
        """将补全后的图像合并回原图像，替换到s14的调整亮度结果中。"""
        original_image = self.get_file_path(self.f17_缩放图像, output_path.stem)
        patched_image = self.get_file_path(self.f22_补全黑线, output_path.stem)
        with Image.open(original_image) as original_image, Image.open(patched_image) as patched_image:
            original_image = np.asarray(original_image, copy=True)
            patched_image = np.asarray(patched_image, copy=True)
        original_image[self.p18_补全时的上下裁剪范围_像素:-self.p18_补全时的上下裁剪范围_像素, :, :] = patched_image
        Image.fromarray(original_image).save(output_path)

    @mark_as_method
    def f24_人工补全黑边(self, output_path: Path) -> None:
        self.get_input_image(self.f23_合并补全图像, output_path).save(output_path)
        raise ManuallyProcessRequiredException


if __name__ == '__main__':
    s25010502_花岗岩的侧面光学扫描的预处理.main()
