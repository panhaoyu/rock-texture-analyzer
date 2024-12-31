import copy

import cv2
import numpy as np
import open3d
from sci_cache import method_cache

from scripts.config import base_dir, project_name
from scripts.p2_调整为主平面 import PointCloudProcessorP2


class PointCloudProcessorP3(PointCloudProcessorP2):
    @property
    @method_cache
    def p3_xOy平面对正(self):
        grid_size = 1
        threshold = 50
        cloud = self.p2_调整为主平面
        cloud = copy.deepcopy(cloud)
        points = np.asarray(cloud.points)
        projected_points = points[:, :2]

        x_min, y_min = projected_points.min(axis=0)
        x_max, y_max = projected_points.max(axis=0)

        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        hist, x_edges, y_edges = np.histogram2d(
            projected_points[:, 0],
            projected_points[:, 1],
            bins=[x_bins, y_bins]
        )

        hist_filtered = np.where(hist > threshold, 255, 0).astype(np.uint8)
        density_image = hist_filtered
        density_image = density_image[::-1]

        contours, _ = cv2.findContours(density_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]

        if angle < -45:
            angle = 90 + angle
        else:
            angle = angle

        theta = np.radians(-angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_z = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        rotated_points = points.dot(R_z.T)
        cloud.points = open3d.utility.Vector3dVector(rotated_points)
        return cloud

    @classmethod
    def main(cls):
        obj = cls(base_dir, project_name)
        obj.绘制点云(obj.p3_xOy平面对正)


if __name__ == '__main__':
    PointCloudProcessorP2.main()
