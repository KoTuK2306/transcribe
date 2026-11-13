"""Форматирование консольного вывода и итогового транскрипта."""

from __future__ import annotations

import os
import sys

# ANSI-коды выключены при перенаправлении вывода или по NO_COLOR (спека
# no-color.org). Проверяется один раз на старте.
_COLOR_ENABLED: bool = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _paint(code: str, text: str) -> str:
    """Обернуть текст в ANSI-последовательность, если цвет разрешён."""
    return f"\033[{code}m{text}\033[0m" if _COLOR_ENABLED else text


def cyan(text: str) -> str:
    return _paint("36;1", text)


def green(text: str) -> str:
    return _paint("32;1", text)


def yellow(text: str) -> str:
    return _paint("33;1", text)


def red(text: str) -> str:
    return _paint("31;1", text)


def dim(text: str) -> str:
    return _paint("2", text)


def bold(text: str) -> str:
    return _paint("1", text)


def fmt_mmss(seconds: float) -> str:
    """Форматировать секунды как m:ss для человекочитаемого вывода."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def fmt_timestamp(seconds: float) -> str:
    """Форматировать секунды как hh:mm:ss — таймкоды в выходном файле."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def stage_header(index: int, title: str, subtitle: str = "") -> None:
    """Печать заголовка этапа и его технических параметров."""
    print()
    print(f"{cyan(f'[{index}/2]')} {bold(title)}")
    if subtitle:
        print(f"        {dim(subtitle)}")


def ok_line(text: str) -> None:
    """Печать строки результата этапа."""
    print(f"        {green('✓')} {text}")


def separator() -> None:
    print()
    print(dim("─" * 60))
