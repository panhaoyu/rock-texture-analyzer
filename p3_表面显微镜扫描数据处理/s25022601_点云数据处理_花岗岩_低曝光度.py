import shutil
import tempfile
from pathlib import Path

import open3d

from p3_表面显微镜扫描数据处理.base import BaseProcessor, mark_as_method, ManuallyProcessRequiredException


class s25022602_劈裂面形貌扫描_花岗岩_低曝光度(BaseProcessor):
    @mark_as_method
    def f1_原始数据(self, output_path: Path) -> None:
        raise ManuallyProcessRequiredException

    @mark_as_method
    def f2_读取点云原始数据(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f1_原始数据, output_path.stem)
        output_path = output_path.with_suffix('.ply')
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_path = Path(tmpdirname) / input_path.name
            shutil.copy2(input_path, temp_path)
            pc = open3d.io.read_point_cloud(temp_path.as_posix())
        open3d.io.write_point_cloud(output_path.as_posix(), pc)


if __name__ == '__main__':
    s25022602_劈裂面形貌扫描_花岗岩_低曝光度.main()
