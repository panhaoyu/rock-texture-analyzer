from pathlib import Path

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

    def f99_处理结果(self, output_path: Path):
        raise ManuallyProcessRequiredException


if __name__ == '__main__':
    Processor.main()
