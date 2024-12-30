import pickle
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Cache:
    """
    一个用于将数据缓存到硬盘上的基类。通过继承此类，子类可以方便地实现数据持久化的功能。
    需要传入一个文件路径来确定缓存文件的位置。在初始化时自动加载缓存（如果存在），并提供手动保存数据的功能。

    Attributes:
        __path (Path): 指定数据缓存的文件路径，必须以 '.pickle' 为后缀。

    Methods:
        save(): 手动保存当前对象的属性到缓存文件中。

    使用方法:
        1. 定义一个继承自 Cache 的数据类，并在其中定义需要持久化的属性:

            @dataclass
            class Data(Cache):
                a: int = None
                b: float = None
                c: str = None
                d: bool = None

        2. 初始化数据类时，传入缓存文件路径，并在需要时调用 `save()` 方法保存数据:

            data = Data(path=Path.home() / 'Desktop/cache.pickle')
            data.a = 213
            data.save()  # 将数据保存到指定的路径

            # 再次初始化时，之前保存的数据将会自动加载
            data = Data(path=Path.home() / 'Desktop/cache.pickle')
            assert data.a == 213

    Raises:
        ValueError: 如果路径后缀不是 '.pickle'，将抛出此异常。
        FileNotFoundError: 如果尝试加载的缓存文件不存在，将抛出此异常。
    """

    __path: Path

    def __post_init__(self):
        if self.__path.suffix != '.pickle':
            raise ValueError("缓存文件必须以 '.pickle' 为后缀")
        self.__load_data()

    def save(self) -> None:
        """
        手动保存当前对象的属性到缓存中。会自动排除掉 `path` 属性。
        """
        data = asdict(self)
        data = {k: v for k, v in data.items() if k != 'path'}
        self.__save(data)

    def __save(self, data: dict) -> None:
        """
        将数据保存到缓存文件中。

        Args:
            data (dict): 需要缓存的数据，以字典形式提供。
        """
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        with self.__path.open('wb') as f:
            pickle.dump(data, f)

    def __load(self) -> dict:
        """
        从缓存文件中加载数据。

        Returns:
            dict: 从缓存加载的数据，以字典形式返回。

        Raises:
            FileNotFoundError: 如果缓存文件不存在，将抛出此异常。
        """
        if self.__path.exists():
            with self.__path.open('rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"缓存文件 {self.__path} 不存在")

    def __load_data(self) -> None:
        """
        从缓存加载数据，并更新当前对象的属性。
        如果缓存文件不存在，则不会更改当前对象的属性。
        """
        try:
            data = self.__load()
            for key, value in data.items():
                setattr(self, key, value)
        except (FileNotFoundError, EOFError):
            pass

    def clear(self):
        self.__path.exists() and self.__path.unlink()
        for k, v in list(self.__dict__.items()):
            if k in ('path',):
                continue
            self.__dict__.pop(k)

        # 要重新实现一些默认值。
        # noinspection PyArgumentList
        self.__init__(self.__path)
        self.__post_init__()
