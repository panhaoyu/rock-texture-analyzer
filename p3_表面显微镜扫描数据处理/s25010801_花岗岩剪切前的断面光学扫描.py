from pathlib import Path

import numpy as np
from PIL import Image

from p3_表面显微镜扫描数据处理.base import BaseProcessor, ManuallyProcessRequiredException


class Processor(BaseProcessor):
    p2_左侧裁剪像素数量 = 2060
    p2_右侧裁剪像素数量 = 1370

    def __init__(self):
        self.base_dir = Path(r'F:\data\laser-scanner\25010801-花岗岩剪切前的断面光学扫描')
        self.source_file_function = self.f1_原始数据
        self.final_file_function = self.f99_处理结果
        self.step_functions = [
            self.f1_原始数据,
            self.f2_左右裁剪,
            self.f99_处理结果,
        ]

    def f1_原始数据(self, output_path: Path):
        raise ManuallyProcessRequiredException

    def f2_左右裁剪(self, output_path: Path):
        input_path = self.get_file_path(self.f1_原始数据, output_path.stem)
        with  Image.open(input_path) as image:
            image = image.copy()
        image = np.asarray(image)
        image = image[:, self.p2_左侧裁剪像素数量:-self.p2_右侧裁剪像素数量]
        image = Image.fromarray(image)
        image.save(output_path)

    def f99_处理结果(self, output_path: Path):
        raise ManuallyProcessRequiredException


if __name__ == '__main__':
    Processor.main()
