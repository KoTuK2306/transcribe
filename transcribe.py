#!/usr/bin/env python3
"""Аудиотранскрипция с диаризацией спикеров.

Конвейер из двух независимых этапов:
    1. Распознавание речи (faster-whisper) -> слова с таймкодами.
    2. Диаризация (pyannote.audio) -> временные интервалы спикеров.

Слова и интервалы сливаются по принципу максимального временного пересечения,
склеиваются в реплики и сохраняются в текстовый файл вида:

    [00:03:15] SPEAKER_00: текст реплики

Модели и результаты каждого этапа кэшируются в --cache-dir: повторный запуск
на том же входе не пересчитывает уже выполненные этапы.

Эксплуатация (установка, токены, примеры запуска) описана в README.md.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import os
import sys
import time

from transcribe.cache import StageCache, cache_key
from transcribe.config import (
    configure_hf_home,
    cuda_available,
    hf_token_from_env,
    setup_environment,
)
from transcribe.output import (
    bold,
    cyan,
    dim,
    fmt_mmss,
    fmt_timestamp,
    green,
    ok_line,
    red,
    separator,
    stage_header,
    yellow,
)
from transcribe.recognizer import transcribe_whisper
from transcribe.diarizer import diarize

# Окружение настраивается до импорта tqdm и любых тяжёлых зависимостей
# (порядок критичен — см. transcribe/config.py).
setup_environment()

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Пороги склейки слов в реплики (см. group_into_lines).
REPLY_MAX_GAP_SEC: float = 2.0
REPLY_MAX_CHARS: int = 60

# ---------------------------------------------------------------------------
# Слияние: слова + интервалы спикеров -> реплики
# ---------------------------------------------------------------------------


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

# ---------------------------------------------------------------------------
# CLI и точка входа
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Транскрипция аудио с разделением по спикерам",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", help="путь к аудиофайлу (mp3/wav/m4a/...)")
    parser.add_argument("--out", default="transcript.txt", help="куда сохранить результат")
    parser.add_argument(
        "--model", default="large-v3",
        help="модель Whisper: tiny/base/small/medium/large-v3",
    )
    parser.add_argument("--language", default=None,
                        help="язык, например ru / en (по умолчанию автоопределение)")
    parser.add_argument("--hf-token", default=hf_token_from_env(),
                        help="токен HuggingFace для pyannote")
    parser.add_argument("--device", default="cuda" if cuda_available() else "cpu",
                        help="cuda или cpu")
    parser.add_argument("--compute-type", default="float16", help="float16 или int8")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="минимальное число спикеров (по умолчанию авто)")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="максимальное число спикеров (по умолчанию авто)")
    parser.add_argument("--cache-dir", default="cache", help="папка кэша")
    parser.add_argument("--no-cache", action="store_true",
                        help="игнорировать кэш и пересчитать всё")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.audio):
        sys.exit(f"{red('Файл не найден:')} {args.audio}")

    configure_hf_home(args.cache_dir)

    device_label = args.device
    if args.device == "cuda" and not cuda_available():
        device_label = yellow("cpu (CUDA недоступна!)")
        args.device = "cpu"

    print()
    print(cyan("  ╔══════════════════════════════════════════════════════╗"))
    print(cyan("  ║") + bold("  Transcribe".ljust(43)[:43] + "  ") + cyan("║"))
    print(cyan("  ╚══════════════════════════════════════════════════════╝"))
    print(f"        файл:     {os.path.basename(args.audio)}")
    print(
        f"        модель:   {args.model} · язык: {args.language or 'авто'}"
        f" · устройство: {device_label}"
    )
    if not args.no_cache:
        print(f"        кэш:      {args.cache_dir}/")

    if args.min_speakers is not None and args.max_speakers is not None \
            and args.min_speakers > args.max_speakers:
        sys.exit(f"{red('Ошибка:')} --min-speakers ({args.min_speakers}) больше "
                 f"--max-speakers ({args.max_speakers})")

    # На CPU compute_type всегда int8 (см. transcribe_whisper), поэтому
    # ключ кэша строим от фактического значения, а не от сырого аргумента —
    # иначе float16/int8 на CPU давали бы два кэша одного результата.
    effective_compute_type = "int8" if args.device == "cpu" else args.compute_type
    cache = StageCache(
        None if args.no_cache else args.cache_dir,
        cache_key(
            args.audio, args.model, args.language, effective_compute_type, args.device,
            args.min_speakers, args.max_speakers,
        ),
    )

    started = time.perf_counter()
    words, time_whisper, duration = transcribe_whisper(
        args.audio, args.model, args.language, args.device,
        args.compute_type, cache,
    )
    turns, time_diarization = diarize(
        args.audio, args.hf_token, args.device, cache,
        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
    )
    time_assign = assign_speakers(words, turns)

    lines = group_into_lines(words)
    total = time.perf_counter() - started

    if not lines:
        sys.exit(f"{red('Ничего не распознано')} — проверьте качество аудио")

    # Нумерация спикеров по убыванию говорливости: SPEAKER_00 — самый
    # активный; человеческие имена подставляются поиском-заменой по файлу.
    # UNKNOWN (слова вне любого интервала диаризации) не переименовываем,
    # чтобы в выводе отличать непривязанные слова от реальных спикеров.
    counts: dict[str, int] = {}
    for line in lines:
        counts[line["speaker"]] = counts.get(line["speaker"], 0) + 1
    order = sorted(counts, key=counts.get, reverse=True)
    rename = {
        sp: (sp if sp == "UNKNOWN" else f"SPEAKER_{i:02d}")
        for i, sp in enumerate(order)
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(f"[{fmt_timestamp(line['start'])}] {rename[line['speaker']]}: {line['text']}\n")

    duration = duration or max(word["end"] for word in words)
    separator()
    print(f"  {green('✓ Готово')}")
    print(f"        результат:    {bold(args.out)}")
    print(
        f"        длительность: {fmt_mmss(duration)} · слов: {len(words)}"
        f" · реплик: {len(lines)}"
    )
    print(f"        спикеры:      {', '.join(rename.values())} ({len(rename)})")
    print(
        f"        время:        распознавание {fmt_mmss(time_whisper)} · "
        f"диаризация {fmt_mmss(time_diarization)} · "
        f"привязка {fmt_mmss(time_assign)} · всего {fmt_mmss(total)}"
    )
    separator()
    print()


if __name__ == "__main__":
    main()