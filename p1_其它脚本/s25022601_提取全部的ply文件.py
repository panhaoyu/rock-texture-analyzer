import shutil
from pathlib import Path

input_dir = Path(r'F:\data\laser-scanner\25022601-劈裂面形貌扫描原始数据\01-劈裂面\立方体-砾岩\低曝光度')
output_dir = Path(r'F:\data\laser-scanner\25022604-劈裂面形貌扫描-砾岩-低曝光度\1-原始数据')

output_dir.mkdir(parents=True, exist_ok=True)
for file in input_dir.glob('*\*.ply'):
    shutil.copy(file, output_dir)
