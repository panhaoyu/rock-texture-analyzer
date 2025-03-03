import shutil
from pathlib import Path

# source_dir = Path(r'F:\data\laser-scanner\25010802-砾岩剪切前的断面光学扫描\99-处理结果')
source_dir = Path(r'F:\data\laser-scanner\25010801-花岗岩剪切前的断面光学扫描\99-处理结果')
output_dir = source_dir / 'output'


def main():
    src: Path
    for src in source_dir.glob('*.png'):
        name = src.stem.replace('-', '')
        name, direction = name[:-1], name[-1]
        direction_dir = output_dir / direction
        direction_dir.mkdir(parents=True, exist_ok=True)
        dst = direction_dir / f'{name}.png'
        shutil.copy(src, dst)


if __name__ == '__main__':
    main()
