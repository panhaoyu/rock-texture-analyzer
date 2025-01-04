import numpy as np
from PIL import Image
from phy_base.scripts.base import PhyCommand

from p2_点云数据处理.config import base_dir, specimen_name


class P2_读取图像(PhyCommand):
    @property
    def raw_image(self) -> np.ndarray:
        path = base_dir / specimen_name / 'optical-image-raw.png'
        return np.asarray(Image.open(path))

    @property
    def cropped(self) -> np.ndarray:
        raise NotImplementedError

    def handle(self, **options):
        path = base_dir / specimen_name / 'optical-image.png'
        Image.fromarray(self.cropped).save(path)


if __name__ == '__main__':
    P2_读取图像.main()
