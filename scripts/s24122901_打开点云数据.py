from pathlib import Path

import more_itertools
import numpy as np
import open3d as o3d

base_dir = Path(r'F:\data\laser-scanner')
project_name = 'Group_4'
ply_file: Path = more_itertools.only(base_dir.glob(f'{project_name}/*.ply'))

# 读取 PLY 点云文件
point_cloud = o3d.io.read_point_cloud(ply_file.as_posix())

# 检查点云数据
print(point_cloud)
print("点云中的点数:", len(point_cloud.points))
print("点云中的颜色数:", len(point_cloud.colors))

# 提取点坐标
points = np.asarray(point_cloud.points)

# 计算点云的质心
centroid = np.mean(points, axis=0)

# 中心化点云
centered_points = points - centroid

# 计算协方差矩阵
cov_matrix = np.cov(centered_points, rowvar=False)

# 进行奇异值分解 (SVD)
_, _, vh = np.linalg.svd(cov_matrix)

# 平面的法向量为最小特征值对应的特征向量
plane_normal = vh[-1]
plane_normal /= np.linalg.norm(plane_normal)

print(f"主体平面法向量: {plane_normal}")

# 目标法向量为z轴
target_normal = np.array([0, 0, 1])

# 计算旋转轴和旋转角度
v = np.cross(plane_normal, target_normal)
s = np.linalg.norm(v)
c = np.dot(plane_normal, target_normal)

if s < 1e-6:
    # 法向量已经对齐，无需旋转
    R = np.eye(3)
else:
    # 计算旋转矩阵
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))

    print(f"旋转矩阵:\n{R}")

# 应用旋转
rotated_points = centered_points.dot(R.T)

# 创建旋转后的点云
point_cloud_rotated = o3d.geometry.PointCloud()
point_cloud_rotated.points = o3d.utility.Vector3dVector(rotated_points)

# 可视化旋转后的点云
o3d.visualization.draw_geometries([point_cloud_rotated],
                                  window_name='旋转后的点云可视化',
                                  width=800,
                                  height=600,
                                  left=50,
                                  top=50,
                                  point_show_normal=False)
