import cv2
import numpy as np
from PIL import Image, ImageCms


def image_diff(
        img1: Image.Image, img2: Image.Image,
        threshold: float = 15,
        kernel_size: int = 3,
) -> np.ndarray:
    """计算两张图像在LAB颜色空间中的显著差异并生成二值差异图。

    通过颜色空间转换、色差计算、阈值处理和形态学操作，识别图像间的视觉显著差异区域。

    Args:
        img1: 输入图像1（PIL Image格式）
        img2: 输入图像2（PIL Image格式）
        threshold: 色差异常判定阈值（默认15.0）
        kernel_size: 形态学操作核尺寸（默认3）

    Returns:
        np.ndarray: 二值化差异图（HxW尺寸，uint8类型，差异区域为1）

    Raises:
        ValueError: 输入图像尺寸不匹配时抛出异常
    """
    if img1.size != img2.size:
        raise ValueError("输入图像尺寸必须一致")

    # LAB颜色空间转换（更符合人眼感知的色差计算）
    srgb_profile, lab_profile = ImageCms.createProfile('sRGB'), ImageCms.createProfile('LAB')
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, 'RGB', 'LAB')
    lab1 = np.asarray(ImageCms.applyTransform(img1, rgb2lab))
    lab2 = np.asarray(ImageCms.applyTransform(img2, rgb2lab))

    # 计算LAB色差欧氏距离并应用阈值
    delta_e = np.sqrt(np.sum((lab1.astype(float) - lab2.astype(float)) ** 2, axis=2))
    binary_diff = (delta_e > threshold).astype(np.uint8)

    # 形态学优化：膨胀连接断裂区域 -> 腐蚀消除孤立噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    optimized_diff = cv2.erode(cv2.dilate(binary_diff, kernel), kernel)

    return optimized_diff.astype(np.uint8)
