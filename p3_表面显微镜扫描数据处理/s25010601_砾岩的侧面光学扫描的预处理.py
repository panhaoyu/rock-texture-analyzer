from pathlib import Path

from p3_表面显微镜扫描数据处理.s25010502_花岗岩的侧面光学扫描的预处理 import s25010502_花岗岩的侧面光学扫描的预处理


class s25010601_砾岩的侧面光学扫描的预处理(s25010502_花岗岩的侧面光学扫描的预处理):
    def __init__(self):
        super().__init__()
        self.base_dir = Path(r'F:\data\laser-scanner\25010601-砾岩的侧面光学扫描的预处理')


if __name__ == '__main__':
    s25010601_砾岩的侧面光学扫描的预处理.main()
