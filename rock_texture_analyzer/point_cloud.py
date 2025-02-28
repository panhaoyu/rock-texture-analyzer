import shutil
import tempfile
import threading
from pathlib import Path

import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


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
        if success := o3d.io.write_point_cloud(temp_file.as_posix(), point_cloud, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_file, output_path)
        return success


# 使用线程本地存储保存每个线程的Visualizer实例
_thread_local = threading.local()


def draw_point_cloud(cloud: Path | PointCloud, output_path: Path) -> None:
    """通过线程本地存储实现无锁可视化"""
    if isinstance(cloud, Path):
        cloud = read_point_cloud(cloud)

    # 每个线程维护自己的Visualizer实例
    if not hasattr(_thread_local, 'vis'):
        _thread_local.vis = o3d.visualization.Visualizer()
        _thread_local.vis.create_window(visible=False)
    else:
        _thread_local.vis.clear_geometries()

    _thread_local.vis.add_geometry(cloud)
    _thread_local.vis.update_geometry(cloud)
    _thread_local.vis.poll_events()
    _thread_local.vis.update_renderer()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / f"temp_{output_path.suffix}"
        _thread_local.vis.capture_screen_image(str(temp_file), do_render=True)
        shutil.copy(temp_file, output_path)
