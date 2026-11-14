"""Этап 1: распознавание речи (faster-whisper) с кэшем результатов."""

from __future__ import annotations

import time

from tqdm import tqdm

from transcribe.cache import StageCache
from transcribe.output import fmt_mmss, ok_line, stage_header


def transcribe_whisper(
    path: str,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
    cache: StageCache,
) -> tuple[list[dict], float, float | None]:
    """Распознать речь и вернуть (слова, длительность_этапа, длительность_аудио).

    Каждое слово — dict {start, end, text} в секундах от начала файла.
    segments из faster-whisper — генератор: материализуем его, попутно
    продвигая прогресс-бар по длительности аудио.
    """
    subtitle = f"{model_size} · {device}"
    if device == "cpu":
        compute_type = "int8"  # float16 на CPU не поддерживается CTranslate2
        subtitle += " (int8; без GPU имеет смысл --model medium)"
    else:
        subtitle += f" ({compute_type})"

    started = time.perf_counter()
    cached = cache.load("whisper")
    if cached is not None:
        # Кэши до версии с длительностью — голый список слов: длительность
        # аудио неизвестна, сводка посчитает её от последнего слова.
        if isinstance(cached, dict):
            words, cached_duration = cached["words"], cached["duration"]
        else:
            words, cached_duration = cached, None
        ok_line(f"Распознавание: кэш, {len(words)} слов")
        return words, time.perf_counter() - started, cached_duration

    from faster_whisper import WhisperModel

    stage_header(1, "Распознавание речи", subtitle)

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        path,
        language=language,
        vad_filter=True,                 # пропускаем участки тишины/шума
        word_timestamps=True,            # обязательно для привязки к спикерам
        beam_size=5,
        condition_on_previous_text=False,  # снижает галлюцинации на шумном входе
    )

    words: list[dict] = []
    with tqdm(
        total=round(info.duration, 1),
        unit="сек",
        desc="        ",
        bar_format="{desc}{percentage:3.0f}% {bar} {n:.0f}/{total:.0f} · {elapsed}<{remaining}",
        ncols=100,
    ) as pbar:
        for segment in segments:
            for word in segment.words or []:
                words.append({"start": word.start, "end": word.end, "text": word.word})
            pbar.update(max(0.0, segment.end - pbar.n))

    elapsed = time.perf_counter() - started
    lang = f"{info.language} ({info.language_probability:.0%})"
    ok_line(f"{len(words)} слов · язык {lang} · {fmt_mmss(elapsed)}")

    # duration сохраняем вместе со словами: без неё повторный запуск по
    # кэшу считал бы длительность аудио от последнего слова, и сводка
    # расходилась бы с первым запуском.
    cache.save("whisper", {"duration": info.duration, "words": words})
    return words, elapsed, info.duration
