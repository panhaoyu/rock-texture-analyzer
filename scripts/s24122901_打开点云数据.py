from pathlib import Path

import more_itertools
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
