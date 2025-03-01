import numpy as np


def export_to_autocad(array: np.ndarray):
    """
    将插值后的高程数据导出为 AutoCAD 支持的 DXF 文件。

    Args:
        array (np.ndarray): 插值后的 [z, r, g, b] 矩阵。
    """
    # 提取并处理高程数据
    resolution_mm = 2.0
    skip_ratio = int(resolution_mm / self.grid_resolution)
    elevation = array[::skip_ratio, ::skip_ratio, 0]
    elevation_min, elevation_max = np.nanquantile(elevation, 0.01), np.nanquantile(elevation, 0.99)
    elevation = np.clip(elevation, elevation_min, elevation_max)

    # 高度放缩系数
    scale_z = 1.0
    avg_z = np.nanmean(elevation)
    elevation = (elevation - avg_z) * scale_z + avg_z

    M, N = elevation.shape
    x_edge = np.linspace(0, (N - 1) * resolution_mm, N)
    y_edge = np.linspace(0, (M - 1) * resolution_mm, M)

    # 创建 DXF 文档
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()

    # Collect vertices and create a vertex map
    vertices = []
    vertex_map = {}
    idx = 0
    for i in range(M):
        for j in range(N):
            z_val = elevation[i, j]
            if not np.isnan(z_val):
                x = x_edge[j]
                y = y_edge[i]
                vertices.append((x, y, z_val))
                vertex_map[(i, j)] = idx
                idx += 1

    # Add vertices and faces using edit_data
    mesh = msp.add_mesh()
    with mesh.edit_data() as mesh_data:
        mesh_data.vertices.extend(vertices)
        for i in range(M - 1):
            for j in range(N - 1):
                if ((i, j) in vertex_map and (i, j + 1) in vertex_map and
                        (i + 1, j) in vertex_map and (i + 1, j + 1) in vertex_map):
                    v0 = vertex_map[(i, j)]
                    v1 = vertex_map[(i, j + 1)]
                    v2 = vertex_map[(i + 1, j)]
                    v3 = vertex_map[(i + 1, j + 1)]
                    mesh_data.faces.append([v0, v1, v2])
                    mesh_data.faces.append([v2, v1, v3])

    # 保存 DXF 文件
    output_dir = self.ply_file.with_name('autocad_exports')
    output_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = output_dir.joinpath('elevation.dxf')
    doc.saveas(str(dxf_path))
    logger.info(f"高程数据已成功导出到 {dxf_path}")
