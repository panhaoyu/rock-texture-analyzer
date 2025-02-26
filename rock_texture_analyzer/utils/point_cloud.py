import shutil
import tempfile
from pathlib import Path

import open3d as o3d


def read_point_cloud(input_path: Path, **kwargs) -> o3d.geometry.PointCloud:
    """通过临时路径读取点云（支持中文路径）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / input_path.name
        shutil.copy2(input_path, temp_file)
        return o3d.io.read_point_cloud(temp_file.as_posix(), **kwargs)


def write_point_cloud(
        output_path: Path,
        point_cloud: o3d.geometry.PointCloud,
        **kwargs
) -> bool:
    """通过临时路径写入点云（支持中文路径）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / f"temp_{output_path.suffix}"
        if (success := o3d.io.write_point_cloud(temp_file.as_posix(), point_cloud, **kwargs)):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_file, output_path)
        return success
