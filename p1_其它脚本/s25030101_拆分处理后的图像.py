import shutil
from pathlib import Path

source_dir = Path(r'F:\data\laser-scanner\25030101-劈裂面形貌扫描\1201-翻转使得上下面可以比较')
suffixes = {'Ua', 'Ub', 'Da', 'Db'}

for f in source_dir.iterdir():
    if not f.is_file():
        continue
    if (suffix := f.stem[-2:]) in suffixes:
        target = source_dir / suffix
        target.mkdir(exist_ok=True)
        shutil.copy(f, target / f'{f.stem[:-2]}{f.suffix}')
