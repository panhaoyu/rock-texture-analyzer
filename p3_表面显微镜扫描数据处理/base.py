import re
import threading
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


class BaseProcessor:
    base_dir: Path
    source_file_function: Callable[[Path], None]
    final_file_function: Callable[[Path], None]
    step_functions: list[Callable[[Path], None]]

    _print_lock: threading.Lock = threading.Lock()

    def print_safe(self, message: str) -> None:
        with self._print_lock:
            print(message)

    def process_stem(self, stem: str) -> None:
        try:
            for func in self.step_functions:
                output_path: Path = self.get_file_path(func, stem)
                if output_path.exists():
                    continue
                func_index, func_name = re.fullmatch(r'f(\d+)_(.*?)', func.__name__).groups()
                func_index = int(func_index)
                try:
                    func(output_path)
                except ManuallyProcessRequiredException as exception:
                    message = exception.args or ()
                    message = ''.join(message)
                    self.print_safe(f'{func_index:02d} {stem:10} {func_name} 需要人工处理：{message}')
                    break
                self.print_safe(f'{func_index:02d} {stem:10} {func_name} 完成')
        except Exception:
            with self._print_lock:
                traceback.print_exc()

    def get_file_path(self, func: Callable[[Path], None], stem: str) -> Path:
        dir_path: Path = self.base_dir / func.__name__.replace('_', '-').lstrip('f')
        extensions: set[str] = {p.suffix for p in dir_path.glob('*') if p.is_file()}
        suffix: str = next(iter(extensions), '.png')
        return dir_path / f'{stem}{suffix}'

    def get_input_path(self, func: Callable[[Path], None], output_path: Path) -> Path:
        return self.get_file_path(func, output_path.stem)

    def get_input_image(self, func: Callable[[Path], None], output_path: Path):
        input_path = self.get_input_path(func, output_path)
        with Image.open(input_path) as img:
            return img.copy()

    def get_input_array(self, func: Callable[[Path], None], output_path: Path):
        path = self.get_input_path(func, output_path)
        match path.suffix:
            case '.png' | '.jpg':
                image = self.get_input_image(func, output_path)
                array = np.asarray(image).copy()
            case '.npy':
                array = np.load(path)
            case other:
                raise NotImplementedError(other)
        return array


    @classmethod
    def main(cls) -> None:
        obj = cls()

        for func in obj.step_functions:
            obj.get_file_path(func, 'dummy').parent.mkdir(parents=True, exist_ok=True)

        source_dir: Path = obj.get_file_path(obj.source_file_function, 'dummy').parent
        stems = [file.stem for file in source_dir.glob('*.jpg')]
        with ThreadPoolExecutor() as executor:
            executor.map(obj.process_stem, stems)
        zip_path = source_dir.parent / f"{source_dir.parent.name}.zip"
        final_dir = obj.get_file_path(obj.final_file_function, 'dummy').parent
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            [zip_file.write(file, file.name) for file in final_dir.glob('*.png')]


class ManuallyProcessRequiredException(Exception):
    pass
