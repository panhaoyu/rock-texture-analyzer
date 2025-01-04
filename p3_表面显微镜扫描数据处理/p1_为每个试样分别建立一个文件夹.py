from pathlib import Path

from PIL import Image

from p3_表面显微镜扫描数据处理 import config


def main():
    i: Path
    for i in config.base_dir.glob('others/1-侧面的光学扫描的原始数据/*.jpg'):
        output_file = config.base_dir / f'{i.stem}' / 'optical-image-raw.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        Image.open(i).save(output_file)


if __name__ == '__main__':
    main()
