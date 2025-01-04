import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


class Processor:
    base_dir = Path(r'F:\data\laser-scanner\others\侧面光学扫描的预处理')
    s1_name = r'1-原始数据'
    s2_name = r'2-转换为PNG'
    s3_name = r'3-裁剪后的PNG'
    print_lock = threading.Lock()

    def __init__(self):
        (self.base_dir / self.s2_name).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.s3_name).mkdir(parents=True, exist_ok=True)

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

    def process_stem(self, stem):
        self.s2_将jpg格式转换为png格式(stem)
        self.s3_裁剪左右两侧(stem)

    @classmethod
    def main(cls):
        obj = cls()
        s1_dir = obj.base_dir / obj.s1_name
        stems = [file.stem for file in s1_dir.glob('*.jpg')]
        with ThreadPoolExecutor() as executor:
            executor.map(obj.process_stem, stems)


if __name__ == '__main__':
    Processor.main()
