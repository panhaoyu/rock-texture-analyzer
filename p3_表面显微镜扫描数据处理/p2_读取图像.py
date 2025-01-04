from functools import cached_property

import cv2
import numpy as np
from PIL import Image
from phy_base.scripts.base import PhyCommand

from p2_点云数据处理.config import base_dir, specimen_name


class P2_读取图像(PhyCommand):
    @property
    def raw_image(self) -> np.ndarray:
        path = base_dir / specimen_name / 'optical-image-raw.png'
        with Image.open(path) as img:
            return np.array(img.convert('RGB'))

    @cached_property
    def hls(self) -> np.ndarray:
        hls = cv2.cvtColor(self.raw_image, cv2.COLOR_RGB2HLS)
        self.save_step(1, '1-hls.png', hls)
        return hls

    @cached_property
    def lightness(self) -> np.ndarray:
        L_channel = self.hls[:, :, 1]
        self.save_step(2, '2-lightness.png', L_channel)
        return L_channel

    @cached_property
    def fixed_threshold(self) -> np.ndarray:
        _, fixed_thresh = cv2.threshold(self.lightness, 15, 255, cv2.THRESH_BINARY)
        self.save_step(3, '3-fixed_threshold.png', fixed_thresh)
        return fixed_thresh

    @cached_property
    def thinned(self) -> np.ndarray:
        thinned = self.thinning(self.fixed_threshold)
        self.save_step(4, '4-thinned.png', thinned)
        return thinned

    def handle(self, **options):
        self.hls
        self.lightness
        self.fixed_threshold
        self.thinned
        path = base_dir / specimen_name / 'optical-image.png'
        Image.fromarray(self.thinned).save(path)

    def save_step(self, step_number: int, filename: str, image: np.ndarray):
        steps_dir = base_dir / specimen_name / 'optical-image-steps'
        step_path = steps_dir / filename
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
        elif len(image.shape) == 2:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_to_save = image
        Image.fromarray(image_to_save).save(step_path)

    def thinning(self, image: np.ndarray) -> np.ndarray:
        size = np.size(image)
        skeleton = np.zeros(image.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        for _ in range(10):
            eroded = cv2.erode(image, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(image, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            image = eroded.copy()

            zeros = size - cv2.countNonZero(image)
            if zeros == size:
                break

        return skeleton


if __name__ == '__main__':
    P2_读取图像.main()
