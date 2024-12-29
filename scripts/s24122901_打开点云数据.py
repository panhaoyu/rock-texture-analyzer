from pathlib import Path

import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import open3d as o3d

# 设置文件路径
base_dir = Path(r'F:\data\laser-scanner')
project_name = 'Group_4'
ply_files = list(base_dir.glob(f'{project_name}/*.ply'))

if not ply_files:
    raise FileNotFoundError(f"No PLY files found in {base_dir / project_name}")

ply_file: Path = more_itertools.only(ply_files)

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

# 定义网格大小
grid_size = 0.1  # 根据需要调整网格大小

# 投影到平面 (z=0)
projected_points = rotated_points[:, :2]

# 计算2D直方图
x_min, y_min = projected_points.min(axis=0)
x_max, y_max = projected_points.max(axis=0)

# 计算网格边界
x_bins = np.arange(x_min, x_max + grid_size, grid_size)
y_bins = np.arange(y_min, y_max + grid_size, grid_size)

# 计算每个网格的点数
hist, x_edges, y_edges = np.histogram2d(projected_points[:, 0],
                                        projected_points[:, 1],
                                        bins=[x_bins, y_bins])

# 打印非零网格点数量
non_zero_cells = np.count_nonzero(hist)
print(f"非零网格点数量: {non_zero_cells}")

# 仅保留数量超过100的网格
threshold = 10
hist_masked = np.where(hist > threshold, hist, np.nan)

# 创建图形
plt.figure(figsize=(10, 8))

# 绘制密度图，只有数量超过阈值的网格会显示颜色
mesh = plt.pcolormesh(x_edges, y_edges, hist_masked.T, cmap='viridis', shading='auto', vmin=threshold)

# 添加颜色条，标签并设置颜色条范围
cbar = plt.colorbar(mesh)

plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')  # 保持比例
plt.tight_layout()
plt.show()
