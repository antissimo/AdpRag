import logging
from logging.handlers import RotatingFileHandler
from .config import LOG_FILE
from pathlib import Path

class FileLogger:
    _instance = None

    def __new__(cls, max_bytes=5*1024*1024, backup_count=1):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

            logger = logging.getLogger("RAGLogger")
            logger.setLevel(logging.INFO)
            logger.propagate = False

            handler = RotatingFileHandler(
                LOG_FILE, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            cls._instance.logger = logger
        return cls._instance

    @classmethod
    def get_logger(cls):
        return cls._instance.logger if cls._instance else cls()

    @classmethod
    def info(cls, msg):
        cls.get_logger().info(msg)

    @classmethod
    def warning(cls, msg):
        cls.get_logger().warning(msg)

    @classmethod
    def error(cls, msg):
        cls.get_logger().error(msg)