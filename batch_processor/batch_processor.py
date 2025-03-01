import re
import threading
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path

from more_itertools import only

from p3_表面显微镜扫描数据处理.config import base_dir
from .processors.base import BaseProcessMethod, ManuallyProcessRequiredException


class BatchProcessor:
    _print_lock: threading.Lock = threading.Lock()

    @classmethod
    def print_safe(cls, message: str) -> None:
        with cls._print_lock:
            print(message)

    @cached_property
    def step_functions(self):
        class_methods: dict[int, list[BaseProcessMethod]] = {}
        for klass in self.__class__.mro():
            for value in vars(klass).values():
                if isinstance(value, BaseProcessMethod):
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
            output_path: Path = func.get_input_path(path)
            if output_path.exists():
                recreate_require = func.is_recreate_required
                if not recreate_require:
                    continue
            func_index, func_name = func.step_index, func.func_name
            func.is_single_thread and func.lock.acquire_lock()
            try:
                result = func(self, output_path)
                func.write(result, output_path)
            except ManuallyProcessRequiredException as exception:
                message = exception.args or ()
                message = ''.join(message)
                self.print_safe(f'{func_index:02d} {stem:10} {func_name} 需要人工处理：{message}')
                break
            except Exception as e:
                self.print_safe(f'{func_index:02d} {stem:10} {func_name} 异常：{e}')
                with self._print_lock:
                    traceback.print_exc()
                break
            finally:
                func.is_single_thread and func.lock.release_lock()
            self.print_safe(f'{func_index:02d} {stem:10} {func_name} 完成')

    enable_multithread: bool = True
    is_debug: bool = False

    @classmethod
    def main(cls) -> None:
        obj = cls()

        functions = obj.step_functions
        source_function = only((f for f in functions if f.is_source), functions[0])
        final_function = only((f for f in functions if f.is_final), functions[-1])

        files = [file for file in source_function.directory.glob(f'*{source_function.suffix}')]
        if cls.is_debug:
            files = files[:2]
        if cls.enable_multithread:
            with ThreadPoolExecutor() as executor:
                executor.map(obj.process_path, files)
        else:
            for file in files:
                obj.process_path(file)
        zip_path = obj.base_dir / f"{obj.base_dir.name}.zip"
        final_dir = final_function.directory
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            [zip_file.write(file, file.name) for file in final_dir.glob('*.png')]
