"""CLI: разбор аргументов и оркестрация этапов конвейера.

Окружение (env/warnings/logging) настраивается точкой входа до импорта
этого модуля — см. setup_environment() в transcribe/config.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from transcribe.aligner import assign_speakers, group_into_lines
from transcribe.cache import StageCache, cache_key
from transcribe.config import configure_hf_home, cuda_available, hf_token_from_env
from transcribe.diarizer import DIARIZATION_MODEL, diarize
from transcribe.output import (
    bold,
    build_speaker_rename,
    cyan,
    print_summary,
    red,
    write_transcript,
    yellow,
)
from transcribe.recognizer import transcribe_whisper


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

    # Ключи кэша строим отдельно для каждого этапа и от идентификаторов
    # моделей: смена модели (Whisper или pyannote) должна инвалидировать
    # только её кэш, а не весь. На CPU compute_type всегда int8
    # (см. transcribe_whisper), поэтому ключ строим от фактического
    # значения, а не от сырого аргумента — иначе float16/int8 на CPU давали
    # бы два кэша одного результата.
    effective_compute_type = "int8" if args.device == "cpu" else args.compute_type
    cache_dir = None if args.no_cache else args.cache_dir
    whisper_cache = StageCache(
        cache_dir,
        cache_key(
            args.audio, "whisper", args.model, args.language,
            effective_compute_type, args.device,
        ),
    )
    diarization_cache = StageCache(
        cache_dir,
        cache_key(
            args.audio, "diarization", DIARIZATION_MODEL, args.device,
            args.min_speakers, args.max_speakers,
        ),
    )

    started = time.perf_counter()
    words, time_whisper, duration = transcribe_whisper(
        args.audio, args.model, args.language, args.device,
        args.compute_type, whisper_cache,
    )
    turns, time_diarization = diarize(
        args.audio, args.hf_token, args.device, diarization_cache,
        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
    )
    time_assign = assign_speakers(words, turns)

    lines = group_into_lines(words)
    total = time.perf_counter() - started

    if not lines:
        sys.exit(f"{red('Ничего не распознано')} — проверьте качество аудио")

    rename = build_speaker_rename(lines)
    write_transcript(args.out, lines, rename)

    duration = duration or max(word["end"] for word in words)
    print_summary(
        out_path=args.out,
        duration=duration,
        words=words,
        lines=lines,
        rename=rename,
        time_whisper=time_whisper,
        time_diarization=time_diarization,
        time_assign=time_assign,
        total=total,
    )
