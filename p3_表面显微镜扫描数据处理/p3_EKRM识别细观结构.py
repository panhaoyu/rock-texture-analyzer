from pathlib import Path

from rock_grain_identifier import RockGrainIdentifier
from rock_grain_identifier.group import RgiGroup


class GraniteIdentifier(RockGrainIdentifier):
    groups = [
        RgiGroup(name='other', index=1, color='#00FF00'),
        RgiGroup(name='feldspar', index=2, color='#FF0000'),
        RgiGroup(name='quartz', index=3, color='#E1FF00'),
        RgiGroup(name='biotite', index=4, color='#000000'),
    ]


def main():
    base_dir: Path = Path(r'F:\data\laser-scanner\others\25010701-花岗岩的细观结构识别')
    identifier = GraniteIdentifier(*base_dir.glob('1-*/*.png'))
    identifier.generate_predict_results()
    # identifier.kmeans_evaluate()
    # identifier.fix_noizy_pixels(pixel_count=1250)  # 噪声处理


if __name__ == '__main__':
    main()
