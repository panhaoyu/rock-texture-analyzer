from pathlib import Path

from PIL import Image


class Processor:
    base_dir = Path(r'F:\data\laser-scanner\others\侧面光学扫描的预处理')
    s1_name = r'1-原始数据'
    s2_name = r'2-裁剪'

    def s1_将jpg格式转换为png格式(self):
        output_dir = self.base_dir / self.s2_name
        for file in self.base_dir.joinpath(self.s1_name).glob('*.jpg'):
            output_file = output_dir.joinpath(f'{file.stem}.png')
            with Image.open(file) as image:
                image.save(output_file)

    @classmethod
    def main(cls):
        obj = cls()
        obj.s1_将jpg格式转换为png格式()


if __name__ == '__main__':
    Processor.main()
