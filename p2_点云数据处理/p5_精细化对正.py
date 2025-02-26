import copy

import numpy as np
import open3d
from sci_cache import sci_method_cache
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from p2_点云数据处理.config import base_dir, project_name
from p2_点云数据处理.p4_调整地面在下方 import PointCloudProcessorP4


class PointCloudProcessorP5(PointCloudProcessorP4):

    @property
    @sci_method_cache
    def p5_优化精细对正(self):

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制点云(obj.p5_优化精细对正)


if __name__ == '__main__':
    PointCloudProcessorP5.main()
