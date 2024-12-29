from pathlib import Path

import more_itertools
import numpy as np
import open3d as o3d

base_dir = Path(r'F:\data\laser-scanner')
project_name = 'Group_3'
ply_file: Path = more_itertools.only(base_dir.glob(f'{project_name}/*.ply'))

# 读取 PLY 点云文件
point_cloud = o3d.io.read_point_cloud(ply_file.as_posix())

# 检查点云数据
print(point_cloud)
print("点云中的点数:", len(point_cloud.points))
print("点云中的颜色数:", len(point_cloud.colors))

# 进行平面分割，使用最小二乘法拟合主体平面
plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
[a, b, c, d] = plane_model
print(f"平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

# 获取平面法向量
plane_normal = np.array([a, b, c])
plane_normal = plane_normal / np.linalg.norm(plane_normal)

# 计算旋转矩阵，使平面法向量对齐到z轴
target_normal = np.array([0, 0, 1])
v = np.cross(plane_normal, target_normal)
if np.linalg.norm(v) < 1e-6:
    R = np.eye(3)
else:
    s = np.linalg.norm(v)
    c_angle = np.dot(plane_normal, target_normal)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c_angle) / (s ** 2))

# 应用旋转
rotated_points = np.asarray(point_cloud.points).dot(R.T)
point_cloud_rotated = o3d.geometry.PointCloud()
point_cloud_rotated.points = o3d.utility.Vector3dVector(rotated_points)

# 可视化旋转后的点云和坐标轴
o3d.visualization.draw_geometries([point_cloud_rotated],
                                  width=800,
                                  height=600,
                                  left=50,
                                  top=50,
                                  point_show_normal=False)
