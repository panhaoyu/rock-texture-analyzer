import shutil

from p2_点云数据处理.config import base_dir


def main():
    for image in base_dir.glob('*/images/*.png'):
        project_name = image.parent.parent.name
        output_dir = base_dir.joinpath('images')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir.joinpath(f'{project_name}-{image.stem}.png')
        shutil.copy(image, output_file)


if __name__ == '__main__':
    main()
