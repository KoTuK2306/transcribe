"""Кэш результатов этапов конвейера.

Ключ — срез SHA-1 от абсолютного пути входа и параметров этапа: любое
изменение входа или конфигурации инвалидирует кэш автоматически.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass


def cache_key(audio_path: str, *parts: object) -> str:
    """Построить ключ кэша из пути к аудио и параметров запуска."""
    digest = hashlib.sha1()
    digest.update(os.path.abspath(audio_path).encode("utf-8"))
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class StageCache:
    """Доступ к кэшу этапов; cache_dir=None отключает кэш (--no-cache)."""

    cache_dir: str | None
    key: str

    def load(self, stage: str) -> list[dict] | None:
        """Прочитать сохранённый результат этапа; None, если кэша нет.

        Битый JSON (прерванная запись) трактуем как отсутствие кэша:
        этап просто пересчитается, вместо падения прогона трейсбэком.
        """
        if self.cache_dir is None:
            return None
        path = os.path.join(self.cache_dir, f"{self.key}.{stage}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def save(self, stage: str, data: list[dict]) -> None:
        """Сохранить результат этапа; каталог создаётся при необходимости."""
        if self.cache_dir is None:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"{self.key}.{stage}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
