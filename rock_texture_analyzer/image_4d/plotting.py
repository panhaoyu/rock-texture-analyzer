import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt, cm


def add_label(image: Image.Image, text: str) -> Image.Image:
    size = image.height // 10
    drawer = ImageDraw.Draw(image)
    font = ImageFont.truetype(r'F:\data\simulation\programs\mpl_data\fonts\ttf\SimSun.ttf', size=size)
    textbox = drawer.textbbox((0, 0), text, font=font)
    drawer.rectangle((5, 5, textbox[2] - textbox[0] + 15, textbox[3] - textbox[1] + 40), fill="white")
    drawer.text((10, 10), text, fill="black", font=font)
    return image


def depth_matrix_to_rgb_image(matrix: np.ndarray, text: str = None) -> Image.Image:
    if len(matrix.shape) == 3 and matrix.shape[2] == 4:
        matrix = matrix[..., 1:]
    assert len(matrix.shape) == 3 and matrix.shape[2] == 3, matrix.shape

    def normalize(channel: np.ndarray) -> np.ndarray:
        v_min = np.nanquantile(channel, 0.01)
        v_max = np.nanquantile(channel, 0.99)
        return np.nan_to_num((channel - v_min) / max(v_max - v_min, 1e-9), copy=False)

    channels = [normalize(matrix[..., i]) * 255 for i in range(3)]
    array = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    text and add_label(image, text)
    return image


def depth_matrix_to_elevation_image(matrix: np.ndarray, v_range=None, text: str = None) -> Image.Image:
    if len(matrix.shape) == 3 and matrix.shape[2] == 4:
        matrix = matrix[..., 0]
    assert len(matrix.shape) == 2 and matrix.shape[0] > 10 and matrix.shape[1] > 10, matrix.shape
    mean = np.nanmean(matrix)
    if v_range is None:
        vmin = np.nanquantile(matrix, 0.01)
        vmax = np.nanquantile(matrix, 0.99)
    else:
        vmin = mean - v_range / 2
        vmax = mean + v_range / 2
    norm = plt.Normalize(vmin, vmax)
    array = (cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(matrix)[..., :3] * 255).astype(np.uint8)
    image = Image.fromarray(array)
    text and add_label(image, text)
    return image


def merge_image_grid(grid: list[list[Image.Image]]) -> Image.Image:
    rows, cols = len(grid), len(grid[0]) if grid else 0
    cell_w, cell_h = grid[0][0].size if rows else (0, 0)
    merged = Image.new('RGB', (cols * cell_w, rows * cell_h))
    [merged.paste(img, (j * cell_w, i * cell_h)) for i, row in enumerate(grid) for j, img in enumerate(row)]
    return merged
