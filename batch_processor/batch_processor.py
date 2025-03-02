import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Type

from more_itertools import only

from p3_表面显微镜扫描数据处理.config import base_dir
from .processors.base import BaseProcessor

logging.basicConfig(
    level=logging.INFO,
    style='{',
    datefmt='%H%M%S',
    format="{levelname:>8} {asctime} {name:<20} {message}"
)
logger = logging.getLogger(Path(__file__).stem)


class BatchManager:
    def __init__(self, klass: Type['SerialProcess']):
        self.klass = klass

    @cached_property
    def step_functions(self):
        class_methods: dict[int, list[BaseProcessor]] = {}
        for klass in self.klass.mro():
            for value in vars(klass).values():
                if isinstance(value, BaseProcessor):
                    class_methods.setdefault(value.step_index, []).append(value)
        sorted_methods = sorted(class_methods.items(), key=lambda x: x[0])
        methods = [method for _, methods in sorted_methods for method in methods]
        for method in methods:
            method.processor = self
        return methods

    @cached_property
    def base_dir(self) -> Path:
        class_name = self.klass.__name__
        match = re.fullmatch(r's(\d{8})_(.+)', class_name)
        if not match:
            raise ValueError(f"无效的类名: {class_name}")
        code, name = match.groups()
        result = base_dir.joinpath(f"{code}-{name}")
        if not result.exists():
            name = name.replace('_', '-')
            result = base_dir.joinpath(f"{code}-{name}")
        assert result.exists(), result
        return result

    def process_path(self, instance: 'SerialProcess') -> None:
        for func in self.step_functions:
            if not func.process(instance):
                break

    enable_multithread: bool = True
    multithread_workers: int = 10
    is_debug: bool = False

    @cached_property
    def source_function(self):
        return only((f for f in self.step_functions if f.is_source), self.step_functions[0])

    @cached_property
    def final_function(self):
        return only((f for f in self.step_functions if f.is_final), self.step_functions[-1])

    @cached_property
    def files(self) -> list[Path]:
        files = [file for file in self.source_function.directory.glob(f'*{self.source_function.suffix}')]
        if self.is_debug:
            files = files[:2]
        return files

    @cached_property
    def stems(self):
        return [file.stem for file in self.files]

    @cached_property
    def instances(self):
        result = [self.klass(file) for file in self.files]
        for instance in result:
            instance.manager = self
        return result

    def main(self) -> None:
        for func in self.step_functions:
            func.all_stems.update(self.stems)
            func.pending_stems.update(self.stems)
        if self.klass.enable_multithread:
            with ThreadPoolExecutor(max_workers=self.klass.multithread_workers) as executor:
                executor.map(self.process_path, self.instances)
        else:
            for instance in self.instances:
                self.process_path(instance)
        zip_path = self.base_dir / f"{self.base_dir.name}.zip"
        final_dir = self.final_function.directory
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            [zip_file.write(file, file.name) for file in final_dir.glob('*.png')]


class SerialProcess:
    manager: BatchManager

    def __init__(self, path: Path):
        self.path = path

    enable_multithread: bool = True
    multithread_workers: int = 10
    is_debug: bool = False

    @classmethod
    def main(cls) -> None:
        BatchManager(cls).main()
