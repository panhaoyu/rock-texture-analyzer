from pathlib import Path

import more_itertools
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

# 可视化点云
o3d.visualization.draw_geometries([point_cloud],
                                  window_name='点云可视化',
                                  width=800,
                                  height=600,
                                  left=50,
                                  top=50,
                                  point_show_normal=False)
