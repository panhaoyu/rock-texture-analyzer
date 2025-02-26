from pathlib import Path

import cv2
import numpy as np
import open3d
from open3d.cpu.pybind.utility import Vector3dVector
from sklearn.cluster import KMeans

from p3_表面显微镜扫描数据处理.base import BaseProcessor, mark_as_method, ManuallyProcessRequiredException, \
    mark_as_single_thread
from rock_texture_analyzer.utils.point_cloud import write_point_cloud, read_point_cloud, draw_point_cloud


class s25022602_劈裂面形貌扫描_花岗岩_低曝光度(BaseProcessor):
    @mark_as_method
    def f1_原始数据(self, output_path: Path) -> None:
        raise ManuallyProcessRequiredException

    @mark_as_method
    def f2_读取点云原始数据(self, output_path: Path) -> None:
        input_path: Path = self.get_file_path(self.f1_原始数据, output_path)
        output_path = output_path.with_suffix('.ply')
        cloud = read_point_cloud(input_path)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f3_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f2_读取点云原始数据, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    def f4_调整为主平面(self, output_path: Path) -> None:
        output_path = output_path.with_suffix('.ply')
        cloud = read_point_cloud(self.get_input_path(self.f2_读取点云原始数据, output_path))
        points = np.asarray(cloud.points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
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
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f5_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f4_调整为主平面, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    def f6_xOy平面对正(self, output_path: Path) -> None:
        output_path = output_path.with_suffix('.ply')
        grid_size = 1
        threshold = 50
        cloud = read_point_cloud(self.get_input_path(self.f4_调整为主平面, output_path))
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
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f7_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f6_xOy平面对正, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

    @mark_as_method
    def f8_调整地面在下(self, output_path: Path) -> None:
        output_path = output_path.with_suffix('.ply')
        cloud = read_point_cloud(self.get_input_path(self.f6_xOy平面对正, output_path))
        points = np.asarray(cloud.points)
        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1].reshape(-1, 1)

        kmeans_x: KMeans = KMeans(n_clusters=2, random_state=0)
        kmeans_x.fit(x)
        centers_x = sorted(kmeans_x.cluster_centers_.flatten())
        xmin, xmax = centers_x[0], centers_x[1]

        kmeans_y: KMeans = KMeans(n_clusters=2, random_state=0)
        kmeans_y.fit(y)
        centers_y = sorted(kmeans_y.cluster_centers_.flatten())
        ymin, ymax = centers_y[0], centers_y[1]

        extend_x = 0.1 * (xmax - xmin)
        extend_y = 0.1 * (ymax - ymin)

        xmin_ext = xmin - extend_x
        xmax_ext = xmax + extend_x
        ymin_ext = ymin - extend_y
        ymax_ext = ymax + extend_y

        boundary_mask = (
                (points[:, 0] >= xmin_ext) & (points[:, 0] <= xmax_ext) &
                (points[:, 1] >= ymin_ext) & (points[:, 1] <= ymax_ext)
        )

        boundary_points = points[boundary_mask]
        external_mask = (
                ((points[:, 0] > xmax_ext) | (points[:, 1] > ymax_ext)) &
                (points[:, 0] >= xmin_ext)
        )
        external_points = points[external_mask]

        if len(boundary_points) == 0 or len(external_points) == 0:
            return

        median_z_inside = np.median(boundary_points[:, 2])
        median_z_outside = np.median(external_points[:, 2])

        if median_z_outside > median_z_inside:
            flipped_points = points.copy()
            flipped_points[:, 2] = -flipped_points[:, 2]
            cloud.points = open3d.utility.Vector3dVector(flipped_points)
        write_point_cloud(output_path, cloud)

    @mark_as_method
    @mark_as_single_thread
    def f9_绘制点云(self, output_path: Path) -> None:
        cloud_path = self.get_file_path(self.f8_调整地面在下, output_path.stem)
        draw_point_cloud(cloud_path, output_path)

if __name__ == '__main__':
    s25022602_劈裂面形貌扫描_花岗岩_低曝光度.main()
