from pathlib import Path

from rock_grain_identifier import RockGrainIdentifier
from rock_grain_identifier.group import RgiGroup

from p3_表面显微镜扫描数据处理.base import BaseProcessor


class GraniteIdentifier(RockGrainIdentifier):
    fix_noize_pixel_count = 1000
    fix_noize_convolve_radius = 10
    groups = [
        RgiGroup(name='other', index=1, color='#00FF00'),
        RgiGroup(name='feldspar', index=2, color='#FF0000'),
        RgiGroup(name='quartz', index=3, color='#E1FF00'),
        RgiGroup(name='biotite', index=4, color='#000000'),
    ]


def main():
    base_dir: Path = Path(r'F:\data\laser-scanner\others\25010701-花岗岩的细观结构识别')
    identifier = GraniteIdentifier(sorted(base_dir.glob('1-*/*.png')))
    # identifier.generate_predict_results()
    # identifier.kmeans_evaluate()
    # identifier.fix_noizy_pixels()
    identifier.predict_all(base_dir / '2-细观结构识别效果')


class s25010701_花岗岩的细观结构识别(BaseProcessor):
    def __init__(self):
        pass


if __name__ == '__main__':
    main()
