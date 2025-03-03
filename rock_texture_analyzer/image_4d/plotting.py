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
    assert matrix.shape[2] == 4, "输入矩阵应为4通道"
    matrix = matrix[:, :, 1:4]

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
    assert matrix.shape[2] == 4
    matrix = matrix[..., 0]
    matrix = matrix - np.mean(matrix)
    v_range = np.nanquantile(matrix, 0.99) if v_range is None else v_range
    norm = plt.Normalize(-v_range, v_range)
    array = (cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(matrix)[..., :3] * 255).astype(np.uint8)
    image = Image.fromarray(array)
    text and add_label(image, text)
    return image
