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
        """
        细化对正，通过分别对X和Y轴进行K-Means聚类，扩展边界范围，并使用SciPy的优化方法旋转优化使四个侧边界与坐标轴对齐。
        """
        cloud = copy.deepcopy(self.p5_优化精细对正)
        points = np.asarray(cloud.points)

        point_z = points[:, 2]
        bottom_center = np.min(point_z)
        top_center = np.max(point_z)
        range_z = (top_center - bottom_center)
        z_selector = (point_z > (bottom_center + range_z * 0.1)) & (point_z < (top_center - range_z * 0.4))
        boundary_points = points[z_selector]
        point_x, point_y = boundary_points[:, 0], boundary_points[:, 1]

        left_center, right_center = get_two_main_value_filtered(point_x)
        front_center, back_center = get_two_main_value_filtered(point_y)

        print(f'{left_center=} {right_center=}')
        print(f'{front_center=} {back_center=}')

        assert back_center > front_center
        assert right_center > left_center

        # 2. 扩展边界范围，向内外分别扩展10%
        extend_x = 0.1 * (right_center - left_center)
        extend_y = 0.1 * (back_center - front_center)

        definite_front = front_center + extend_y
        definite_back = back_center - extend_y
        definite_left = left_center + extend_x
        definite_right = right_center - extend_x

        left_points = boundary_points[np.abs(point_x - left_center) < extend_x]
        right_points = boundary_points[np.abs(point_x - right_center) < extend_x]
        front_points = boundary_points[np.abs(point_y - front_center) < extend_y]
        back_points = boundary_points[np.abs(point_y - back_center) < extend_y]

        left_points = left_points[(left_points[:, 1] > definite_front) & (left_points[:, 1] < definite_back)]
        right_points = right_points[(right_points[:, 1] > definite_front) & (right_points[:, 1] < definite_back)]
        front_points = front_points[(front_points[:, 0] > definite_left) & (front_points[:, 0] < definite_right)]
        back_points = back_points[(back_points[:, 0] > definite_left) & (back_points[:, 0] < definite_right)]

        left_mean = np.mean(left_points[:, 0])
        left_std = np.std(left_points[:, 0])
        right_mean = np.mean(right_points[:, 0])
        right_std = np.std(right_points[:, 0])
        front_mean = np.mean(front_points[:, 1])
        front_std = np.std(front_points[:, 1])
        back_mean = np.mean(back_points[:, 1])
        back_std = np.std(back_points[:, 1])

        std_range = 5
        left = left_mean + std_range * left_std
        right = right_mean - std_range * right_std
        front = front_mean + std_range * front_std
        back = back_mean - std_range * back_std

        print(f'{left=} {right=} {front=} {back=}')

        point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]
        top_selector = (
                (point_x > left) & (point_x < right)
                & (point_y > front) & (point_y < back)
                & (point_z > bottom_center + range_z * 0.5)
        )

        cloud.points = open3d.utility.Vector3dVector(points[top_selector])

        colors = np.asarray(cloud.colors)  # 扫描的时候未必开启对颜色的扫描
        if colors.size:
            cloud.colors = open3d.utility.Vector3dVector(colors[top_selector])
        return cloud

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制点云(obj.p6_仅保留顶面)


if __name__ == '__main__':
    PointCloudProcessorP6.main()
