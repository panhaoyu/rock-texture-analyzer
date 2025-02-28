from functools import cached_property
from pathlib import Path
from typing import Type

from rock_grain_identifier import RockGrainIdentifier
from rock_grain_identifier.group import RgiGroup

from rock_texture_analyzer.base import BaseProcessor, mark_as_png, ManuallyProcessRequiredException, \
    mark_as_single_thread


class GraniteIdentifier(RockGrainIdentifier):
    fix_noize_pixel_count = 1000
    fix_noize_convolve_radius = 10
    groups = [
        RgiGroup(name='other', index=1, color='#00FF00'),
        RgiGroup(name='feldspar', index=2, color='#FF0000'),
        RgiGroup(name='quartz', index=3, color='#E1FF00'),
        RgiGroup(name='biotite', index=4, color='#000000'),
    ]


class s25010701_花岗岩的细观结构识别(BaseProcessor):
    @mark_as_png
    def f1_原始图像(self, output_path: Path):
        raise ManuallyProcessRequiredException

    identifier_class: Type[GraniteIdentifier] = GraniteIdentifier

    @cached_property
    def identifier(self):
        files = sorted(self.base_dir.glob('1-*/*.png'))
        identifier = self.identifier_class(files)
        return identifier

    @mark_as_single_thread
    def f2_聚类(self, output_path: Path):
        output_path.touch()
        if not list(output_path.parent.glob(f'*{output_path.suffix}')):
            self.identifier.generate_predict_results()
        raise ManuallyProcessRequiredException('需要人工完成聚类')

    @mark_as_single_thread
    def f3_评估(self, output_path: Path):
        output_path.touch()
        if not list(output_path.parent.glob(f'*{output_path.suffix}')):
            self.identifier.kmeans_evaluate()
        raise ManuallyProcessRequiredException('需要检查评估效果')

    @mark_as_single_thread
    def f4_噪声处理(self, output_path: Path):
        output_path.touch()
        if not list(output_path.parent.glob(f'*{output_path.suffix}')):
            self.identifier.fix_noizy_pixels()
        raise ManuallyProcessRequiredException('需要检查噪声处理效果')

    @mark_as_single_thread
    def f5_全部处理(self, output_path: Path):
        if not list(output_path.parent.glob(f'*{output_path.suffix}')):
            with self.identifier.skip_saving_numpy(), self.identifier.skip_saving_thumbnail():
                self.identifier.predict_all(self.base_dir / '2-处理效果/output')
        input_path = output_path.parent.joinpath(f'output/{output_path.stem}-3-fixed.png')
        output_path.hardlink_to(input_path)


if __name__ == '__main__':
    s25010701_花岗岩的细观结构识别.main()
