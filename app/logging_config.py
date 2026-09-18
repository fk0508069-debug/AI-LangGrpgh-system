"""One logging config, called once from api.py / main.py."""

from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_NOISY = (
    "httpx", "httpcore", "urllib3", "chromadb", "pymongo",
    "sentence_transformers", "transformers", "huggingface_hub",
    "filelock", "asyncio", "multipart", "faiss", "faiss.loader",
)


def setup_logging(level: str | None = None) -> None:
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(resolved)
    else:
        logging.basicConfig(
            level=resolved,
            format=_FMT,
            datefmt=_DATEFMT,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)