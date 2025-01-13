from pathlib import Path

from rock_grain_identifier import RockGrainIdentifier
from rock_grain_identifier.group import RgiGroup

from p3_表面显微镜扫描数据处理.base import BaseProcessor, mark_as_method, ManuallyProcessRequiredException


class s25010701_花岗岩的细观结构识别(BaseProcessor):
    @mark_as_method
    def f1_原始图像(self, output_path: Path):
        raise ManuallyProcessRequiredException

    @mark_as_method
    def f2_处理效果(self, output_path: Path):
        input_path = output_path.parent.joinpath(f'output/{output_path.stem}-3-fixed.png')
        output_path.hardlink_to(input_path)

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
    obj = s25010701_花岗岩的细观结构识别()
    identifier = GraniteIdentifier(sorted(obj.base_dir.glob('1-*/*.png')))
    # identifier.generate_predict_results()
    # identifier.kmeans_evaluate()
    # identifier.fix_noizy_pixels()
    # with identifier.skip_saving_numpy(), identifier.skip_saving_thumbnail():
    #     identifier.predict_all(obj.base_dir / '2-处理效果/output')

    s25010701_花岗岩的细观结构识别.main()

if __name__ == '__main__':
    main()
