from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


class Processor:
    base_dir = Path(r'F:\data\laser-scanner\others\侧面光学扫描的预处理')
    s1_name = r'1-原始数据'
    s2_name = r'2-转换为PNG'
    s3_name = r'3-裁剪后的PNG'

    def s2_将jpg格式转换为png格式(self):
        output_dir = self.base_dir / self.s2_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in self.base_dir.joinpath(self.s1_name).glob('*.jpg'):
            output_file = output_dir.joinpath(f'{file.stem}.png')
            with Image.open(file) as image:
                image.save(output_file)
                print(f"已转换并保存: {output_file}")

    def s3_裁剪左右两侧(self):
        input_dir = self.base_dir / self.s2_name
        output_dir = self.base_dir / self.s3_name
        output_dir.mkdir(parents=True, exist_ok=True)
        files = list(input_dir.glob('*.png'))
        left_crop = 1600
        right_crop = 1200

        def process(file_path):
            output_file = output_dir / f"{file_path.stem}_cropped.png"
            with Image.open(file_path) as image:
                width, height = image.size
                if width <= left_crop + right_crop:
                    print(f"图像 {file_path.name} 宽度不足以截取 {left_crop} 左边和 {right_crop} 右边像素。跳过处理。")
                    return
                cropped_image = image.crop((left_crop, 0, width - right_crop, height))
                cropped_image.save(output_file)
                print(f"已裁剪并保存: {output_file}")

        with ThreadPoolExecutor() as executor:
            executor.map(process, files)

    @classmethod
    def main(cls):
        obj = cls()
        # obj.s2_将jpg格式转换为png格式()
        obj.s3_裁剪左右两侧()

if __name__ == '__main__':
    Processor.main()
