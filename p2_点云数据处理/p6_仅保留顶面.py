import copy

import numpy as np
import open3d
from sci_cache import sci_method_cache

from p2_点云数据处理.config import base_dir, project_name
from p2_点云数据处理.p5_精细化对正 import PointCloudProcessorP5
from rock_texture_analyzer.utils.get_two_peaks import get_two_main_value_filtered


class PointCloudProcessorP6(PointCloudProcessorP5):
    @property
    @sci_method_cache
    def p6_仅保留顶面(self):
        return cloud

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制点云(obj.p6_仅保留顶面)


if __name__ == '__main__':
    PointCloudProcessorP6.main()
