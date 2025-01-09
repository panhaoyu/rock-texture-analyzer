from pathlib import Path

from p3_表面显微镜扫描数据处理.base import BaseProcessor, ManuallyProcessRequiredException


class Processor(BaseProcessor):
    def __init__(self):
        self.base_dir = Path(r'F:\data\laser-scanner\砾岩临时处理文件夹')
        self.source_file_function = self.f1_原始数据
        self.final_file_function = self.f99_处理结果
        self.step_functions = [
            self.f1_原始数据,
            self.f99_处理结果,
        ]

    def f1_原始数据(self, output_path: Path):
        raise ManuallyProcessRequiredException

    def f99_处理结果(self, output_path: Path):
        raise ManuallyProcessRequiredException


if __name__ == '__main__':
    Processor.main()
