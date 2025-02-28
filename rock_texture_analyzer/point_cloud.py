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


_thread_local = threading.Lock()


def draw_point_cloud(cloud: Path | PointCloud, output_path: Path) -> None:
    """通过临时路径保存点云可视化截图（支持中文路径）"""
    if isinstance(cloud, Path):
        cloud = read_point_cloud(cloud)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / f"temp_{output_path.suffix}"
        with _thread_local:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.clear_geometries()
            vis.add_geometry(cloud)
            vis.update_geometry(cloud)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(temp_file.as_posix(), do_render=True)
            vis.destroy_window()
        shutil.copy2(temp_file, output_path)

