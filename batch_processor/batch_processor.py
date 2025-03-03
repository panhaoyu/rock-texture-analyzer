import logging
from pathlib import Path

from .manager import BatchManager

logging.basicConfig(
    level=logging.INFO,
    style='{',
    datefmt='%H%M%S',
    format="{levelname:>8} {asctime} {name:<20} {message}"
)


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
