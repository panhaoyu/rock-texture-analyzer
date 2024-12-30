import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


def get_two_main_value_filtered(data: np.ndarray, bin_width: int = 50, peak_prominence: float = 1000) -> \
        tuple[float, float]:
    """
    去除背景数据后提取双峰分布的两个主要峰值.

    Args:
        data (np.ndarray): 输入数据.
        bin_width (int): 直方图的 bin 宽度.
        peak_prominence (float): 用于识别峰值的显著性阈值.

    Returns:
        tuple[float, float]: 两个主要峰值的中心位置.
    """
    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 获取 bin 中心位置

    # 检测直方图中的峰值
    peaks, _ = find_peaks(hist, prominence=peak_prominence)
    if len(peaks) < 2:
        raise ValueError("无法检测到两个明显的峰值，请调整参数.")

    # 获取峰的范围（每个峰附近的区域）
    peak_ranges = []
    for peak in peaks:
        left = max(0, peak - 5)  # 峰左侧范围
        right = min(len(hist) - 1, peak + 5)  # 峰右侧范围
        peak_ranges.append((bin_centers[left], bin_centers[right]))

    # 筛选数据，仅保留在两个峰附近的数据
    filtered_data = data[np.any([(data >= r[0]) & (data <= r[1]) for r in peak_ranges], axis=0)]

    # 用高斯混合模型拟合筛选后的数据
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(filtered_data.reshape(-1, 1))

    # 提取两个高斯分布的均值
    means = gmm.means_.flatten()

    # 返回按大小排序的两个峰值
    return tuple(sorted(means))
