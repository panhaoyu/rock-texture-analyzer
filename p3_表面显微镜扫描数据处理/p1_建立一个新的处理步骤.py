import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


class Processor:
    base_dir = Path(r'F:\data\laser-scanner\others\侧面光学扫描的预处理')
    s1_name = r'1-原始数据'
    s2_name = r'2-转换为PNG'
    s3_name = r'3-裁剪后的PNG'
    s4_name = r'4-灰度图像'
    s5_name = r'5-二值化图像'
    print_lock = threading.Lock()

    def __init__(self):
        (self.base_dir / self.s2_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s3_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s4_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s5_name).mkdir(parents=True, exist_ok=True)

    def print_safe(self, message):
        with self.print_lock:
            print(message)

    def s2_将jpg格式转换为png格式(self, stem):
        input_file = self.base_dir / self.s1_name / f"{stem}.jpg"
        output_file = self.base_dir / self.s2_name / f"{stem}.png"
        if output_file.exists():
            self.print_safe(f"{stem} 已存在，跳过转换。")
            return
        with Image.open(input_file) as image:
            image.save(output_file)
        self.print_safe(f"{stem} 已转换并保存。")

    def s3_裁剪左右两侧(self, stem):
        input_file = self.base_dir / self.s2_name / f"{stem}.png"
        output_file = self.base_dir / self.s3_name / f"{stem}.png"
        if output_file.exists():
            self.print_safe(f"{stem} 已存在，跳过裁剪。")
            return
        with Image.open(input_file) as image:
            left_crop = 1600
            right_crop = 1200
            width, height = image.size
            if width <= left_crop + right_crop:
                self.print_safe(f"{stem} 图像宽度不足以截取 {left_crop} 左边和 {right_crop} 右边像素。跳过裁剪。")
                return
            cropped_image = image.crop((left_crop, 0, width - right_crop, height))
            cropped_image.save(output_file)
        self.print_safe(f"{stem} 已裁剪并保存。")

    def s4_转换为灰度(self, stem):
        input_file = self.base_dir / self.s3_name / f"{stem}.png"
        output_file = self.base_dir / self.s4_name / f"{stem}.png"
        if output_file.exists():
            self.print_safe(f"{stem} 已存在，跳过灰度转换。")
            return
        with Image.open(input_file) as image:
            grayscale_image = image.convert("L")
            grayscale_image.save(output_file)
        self.print_safe(f"{stem} 已转换为灰度。")

    def s5_二值化(self, stem):
        input_file = self.base_dir / self.s4_name / f"{stem}.png"
        output_file = self.base_dir / self.s5_name / f"{stem}.png"
        if output_file.exists():
            self.print_safe(f"{stem} 已存在，跳过二值化。")
            return
        with Image.open(input_file) as image:
            histogram = image.histogram()
            # 简单双峰检测
            peak1 = histogram.index(max(histogram))
            histogram[peak1] = 0
            peak2 = histogram.index(max(histogram))
            if peak1 < peak2:
                start, end = peak1, peak2
            else:
                start, end = peak2, peak1
            valley = min(range(start, end + 1), key=lambda x: histogram[x])
            threshold = valley
            binary_image = image.point(lambda p: 255 if p > threshold else 0)
            binary_image.save(output_file)
        self.print_safe(f"{stem} 已二值化并保存。")

    def process_stem(self, stem):
        self.s2_将jpg格式转换为png格式(stem)
        self.s3_裁剪左右两侧(stem)
        self.s4_转换为灰度(stem)
        # self.s5_二值化(stem)

    @classmethod
    def main(cls):
        obj = cls()
        s1_dir = obj.base_dir / obj.s1_name
        stems = [file.stem for file in s1_dir.glob('*.jpg')]
        with ThreadPoolExecutor() as executor:
            executor.map(obj.process_stem, stems)


if __name__ == '__main__':
    Processor.main()
