"""Слияние: привязка слов к спикерам и склейка слов в реплики."""

from __future__ import annotations

import time
from bisect import bisect_right

from tqdm import tqdm

# Пороги склейки слов в реплики (см. group_into_lines).
REPLY_MAX_GAP_SEC: float = 2.0
REPLY_MAX_CHARS: int = 60


def assign_speakers(words: list[dict], speaker_turns: list[dict]) -> float:
    """Назначить каждому слову спикера по максимальному временному пересечению.

    Слова и интервалы из разных моделей слегка расходятся по границам,
    поэтому берётся не «спикер интервала, в который попал старт слова»,
    а максимум пересечения — это устойчиво к смещениям до ~сотен мс.
    Слова вне любого интервала остаются UNKNOWN.

    Интервалы одного спикера не пересекаются, поэтому кандидатов на
    пересечение с любым словом — единицы. Двоичный поиск по стартам
    интервалов + обратный проход до первого непересекающегося даёт
    линейное суммарное время вместо квадратичного (слово × интервалы).
    """
    started = time.perf_counter()
    turns = sorted(speaker_turns, key=lambda turn: turn["start"])
    turn_starts = [turn["start"] for turn in turns]
    for word in tqdm(
        words,
        desc="        привязка слов",
        bar_format="{desc} {percentage:3.0f}% {bar} {n}/{total} · {elapsed}",
        ncols=100,
    ):
        best, best_overlap = "UNKNOWN", 0.0
        # turns с start <= word.end; дальше в списке пересечение невозможно.
        candidate = bisect_right(turn_starts, word["end"]) - 1
        while candidate >= 0 and turns[candidate]["end"] > word["start"]:
            turn = turns[candidate]
            overlap = min(word["end"], turn["end"]) - max(word["start"], turn["start"])
            if overlap > best_overlap:
                best, best_overlap = turn["speaker"], overlap
            candidate -= 1
        word["speaker"] = best
    return time.perf_counter() - started


def group_into_lines(
    words: list[dict],
    max_gap: float = REPLY_MAX_GAP_SEC,
    max_len: int = REPLY_MAX_CHARS,
) -> list[dict]:
    """Склеить слова в реплики.

    Новая реплика начинается при смене спикера или паузе длиннее max_gap.
    Длинные реплики принудительно разбиваются по max_len символов, чтобы
    строки в файле оставались читабельными.
    """
    lines: list[dict] = []
    current: dict | None = None
    for word in words:
        new_speaker = current is None or word["speaker"] != current["speaker"]
        gap = current is not None and (word["start"] - current["end"]) > max_gap
        if new_speaker or gap:
            if current:
                lines.append(current)
            current = {
                "speaker": word["speaker"],
                "start": word["start"],
                "end": word["end"],
                "text": word["text"].strip(),
            }
        else:
            current["text"] += " " + word["text"].strip()
            current["end"] = word["end"]
        if current and len(current["text"]) > max_len:
            lines.append(current)
            current = None
    if current:
        lines.append(current)
    return lines
