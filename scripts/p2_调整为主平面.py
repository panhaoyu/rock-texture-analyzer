import copy

import numpy as np
from open3d.cpu.pybind.utility import Vector3dVector
from sci_cache import method_cache

from scripts.config import base_dir, project_name
from scripts.p1_读取点云数据 import PointCloudProcessor


class PointCloudProcessorP2(PointCloudProcessor):

    @property
    @method_cache
    def p2_调整为主平面(self):
        cloud = self.p1_读取点云原始数据
        points = np.asarray(cloud.points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        cloud = copy.deepcopy(cloud)
        cloud.points = Vector3dVector(centered_points)

        cov_matrix = np.cov(centered_points, rowvar=False)
        _, _, vh = np.linalg.svd(cov_matrix)
        plane_normal = vh[-1]
        plane_normal /= np.linalg.norm(plane_normal)

        target_normal = np.array([0, 0, 1])
        v = np.cross(plane_normal, target_normal)
        s = np.linalg.norm(v)
        c = np.dot(plane_normal, target_normal)

        if s < 1e-6:
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))

        rotated_points = centered_points.dot(R.T)
        cloud.points = Vector3dVector(rotated_points)
        return cloud

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制点云(obj.p2_调整为主平面)


if __name__ == '__main__':
    PointCloudProcessorP2.main()
