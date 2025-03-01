import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path

from more_itertools import only

from p3_表面显微镜扫描数据处理.config import base_dir
from .processors.base import BaseProcessor, ManuallyProcessRequiredException

logger = logging.getLogger('BatchProcessor')


class BatchProcessor:
    @cached_property
    def step_functions(self):
        class_methods: dict[int, list[BaseProcessor]] = {}
        for klass in self.__class__.mro():
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
        class_name = self.__class__.__name__
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

    def process_path(self, path: Path) -> None:
        stem = path.stem
        for func in self.step_functions:
            path: Path = func.get_input_path(path)
            if path.exists():
                recreate_require = func.is_recreate_required
                if not recreate_require:
                    continue
            func_index, func_name = func.step_index, func.func_name
            func.is_single_thread and func.single_thread_process_lock.acquire_lock()
            with func.meta_process_lock:
                func.pending_stems.remove(stem)
                func.processing_stems.add(stem)
                if not func.processing_stems and not func.processed_stems:
                    func.on_batch_started()
            try:
                result = func(self, path)
                func.write(result, path)
            except ManuallyProcessRequiredException as exception:
                message = exception.args or ()
                message = ''.join(message)
                logger.info(f'{func_index:02d} {stem:10} {func_name} 需要人工处理：{message}')
                break
            except Exception as e:
                logger.exception(f'{func_index:02d} {stem:10} {func_name} 异常：{e}')
                break
            finally:
                func.is_single_thread and func.single_thread_process_lock.release_lock()
            logger.info(f'{func_index:02d} {stem:10} {func_name} 完成')
            with func.meta_process_lock:
                func.processing_stems.remove(stem)
                func.processed_stems.add(stem)
                if not func.pending_stems and not func.processing_stems:
                    func.on_batch_finished()

    enable_multithread: bool = True
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

    @classmethod
    def main(cls) -> None:
        obj = cls()
        all_stems = [file.stem for file in obj.files]
        for func in obj.step_functions:
            func.all_stems.update(all_stems)
            func.pending_stems.update(all_stems)
        if cls.enable_multithread:
            with ThreadPoolExecutor() as executor:
                executor.map(obj.process_path, obj.files)
        else:
            for file in obj.files:
                obj.process_path(file)
        zip_path = obj.base_dir / f"{obj.base_dir.name}.zip"
        final_dir = obj.final_function.directory
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            [zip_file.write(file, file.name) for file in final_dir.glob('*.png')]
