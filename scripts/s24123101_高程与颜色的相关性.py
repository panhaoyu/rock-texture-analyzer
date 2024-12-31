import numpy as np

from scripts.config import base_dir, project_name
from scripts.s24122901_打开点云数据 import PointCloudProcessor

np.set_printoptions(linewidth=400, precision=3, edgeitems=6)
array = PointCloudProcessor(base_dir, project_name).p7_表面二维重构

slicing = 1
# array = array[slicing:-slicing, slicing:-slicing, :]
print(array[:, :, 0])
print(array[:, :, 1])
print(array[:, :, 2])
print(array[:, :, 3])
print(array.size)
print(np.isnan(array).sum())
print(np.isnan(array).sum() / array.size)
