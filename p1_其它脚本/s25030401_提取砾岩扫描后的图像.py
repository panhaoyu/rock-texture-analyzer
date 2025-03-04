import shutil
from pathlib import Path

source_dir = Path(r'F:\data\laser-scanner\25010802-砾岩剪切前的断面光学扫描\18-调整方向')
# source_dir = Path(r'F:\data\laser-scanner\25010801-花岗岩剪切前的断面光学扫描\18-调整方向')
# source_dir = Path(r'F:\data\laser-scanner\25010601-砾岩的侧面光学扫描的预处理\24-人工补全黑边')
# source_dir = Path(r'F:\data\laser-scanner\25010502-花岗岩的侧面光学扫描的预处理\24-人工补全黑边')
output_dir = source_dir / 'output'

拆分方向 = True

def main():
    src: Path
    for src in source_dir.glob('*.png'):
        name = src.stem.replace('-', '')
        final_output_dir = output_dir
        if 拆分方向:
            name, direction = name[:-1], name[-1]
            final_output_dir = final_output_dir / direction
        final_output_dir.mkdir(parents=True, exist_ok=True)
        dst = final_output_dir / f'{name}.png'
        shutil.copy(src, dst)


if __name__ == '__main__':
    main()
