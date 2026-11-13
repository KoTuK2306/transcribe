"""Окружение и конфигурация: переменные среды, CUDA, токен HuggingFace.

Никакого состояния — только функции настройки и чтения окружения.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

# Имя переменной окружения с токеном HuggingFace (читается также
# transcribe_command.sh из .env).
HF_TOKEN_ENV = "HF_TOKEN"


def setup_environment() -> None:
    """Глушим шум тяжёлых зависимостей ДО их импорта.

    Порядок критичен: huggingface_hub читает HF_HUB_DISABLE_PROGRESS_BARS при
    импорте, логгеры torch создаются при первом обращении, поэтому всё, что
    можно выключить молча, выключается здесь, а не рядом с местом использования.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Консоль Windows по умолчанию в cp1252/ cp1251: кириллица в print и
    # tqdm крашит прогон с UnicodeEncodeError. Принудительно переводим потоки
    # вывода в UTF-8 — это ничего не ломает при перенаправлении в файл.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III subtype is unknown.*")
    warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")

    for logger_name in (
        "torch.utils.flop_counter",  # "triton not found; flop counting..." на Windows
        "torch._dynamo",
        "torch._inductor",
        "torch.cuda.graphs",
        "lightning_fabric",
        "lightning",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def configure_hf_home(cache_dir: str) -> None:
    """Направить кэш моделей HuggingFace в cache_dir/hf.

    HF_HOME читается huggingface_hub при импорте, поэтому задаём до
    импорта любых зависимостей, использующих его (Whisper, pyannote).
    """
    os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(cache_dir, "hf")))


def cuda_available() -> bool:
    """True, если torch видит CUDA-устройство. Импорт внутри функции,
    чтобы не тянуть torch ради одной проверки."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def hf_token_from_env() -> str | None:
    """Токен HuggingFace из переменной окружения (HF_TOKEN)."""
    return os.environ.get(HF_TOKEN_ENV)
