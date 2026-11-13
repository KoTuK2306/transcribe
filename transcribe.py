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
import hashlib
from bisect import bisect_right
import inspect
import json
import logging
import os
import sys
import time
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Окружение: глушим шум тяжёлых зависимостей ДО их импорта.
#
# Порядок критичен: huggingface_hub читает HF_HUB_DISABLE_PROGRESS_BARS при
# импорте, логгеры torch создаются при первом обращении, поэтому всё, что
# можно выключить молча, выключается здесь, а не рядом с местом использования.
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Консоль Windows по умолчанию в cp1252/ cp1251: кириллица в print и
# tqdm крашит прогон с UnicodeEncodeError. Принудительно переводим потоки
# вывода в UTF-8 — это ничего не ломает при перенаправлении в файл.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III subtype is unknown.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")

for _logger_name in (
    "torch.utils.flop_counter",  # "triton not found; flop counting..." на Windows
    "torch._dynamo",
    "torch._inductor",
    "torch.cuda.graphs",
    "lightning_fabric",
    "lightning",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Кандидаты на расположение bin-директории ffmpeg. Используется только на
# Windows и только для связки pyannote.audio 4.x + torchcodec; pyannote 3.x
# декодирует через soundfile и ffmpeg не требует.
FFMPEG_BIN_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files\ffmpeg\bin",
    r"C:\ffmpeg\bin",
)

# Пороги склейки слов в реплики (см. group_into_lines).
REPLY_MAX_GAP_SEC: float = 2.0
REPLY_MAX_CHARS: int = 60

# ANSI-коды выключены при перенаправлении вывода или по NO_COLOR (спека
# no-color.org). Проверяется один раз на старте.
_COLOR_ENABLED: bool = sys.stdout.isatty() and "NO_COLOR" not in os.environ

# ---------------------------------------------------------------------------
# Форматирование консольного вывода
# ---------------------------------------------------------------------------


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

# ---------------------------------------------------------------------------
# Совместимость платформ и версий
#
# Каждый патч здесь — обходной путь для конкретного конфликта версий; если
# зависимости обновились и конфликт исчез, патч должен стать безопасной no-op.
# ---------------------------------------------------------------------------


def register_ffmpeg_dll_dirs() -> None:
    """Зарегистрировать каталоги с DLL ffmpeg для загрузчика Windows.

    ctypes/os.add_dll_directory не используют PATH оболочки, поэтому при
    pyannote.audio 4.x (декодирование через torchcodec) ffmpeg.exe «в PATH»
    недостаточно — нужен именно add_dll_directory к папке с avcodec/avformat.
    Папка из PATH тоже подходит: ищем ffmpeg через shutil.which.
    """
    if sys.platform != "win32":
        return
    import shutil

    candidates = list(FFMPEG_BIN_CANDIDATES)
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        candidates.insert(0, os.path.dirname(ffmpeg_exe))
    for path in candidates:
        if os.path.isdir(path):
            os.add_dll_directory(path)


def patch_torchaudio_for_pyannote3() -> None:
    """Вернуть torchaudio.AudioMetaData, удалённый в torchaudio >= 2.10.

    pyannote.audio 3.x ссылается на атрибут в аннотации возврата io.Audio.__call__
    (вычисляется в момент определения функции), поэтому отсутствие атрибута
    убивает импорт всего пакета. Заглушки достаточно: аннотация не влияет
    на поведение.
    """
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = type("AudioMetaData", (), {})


def patch_torch_load_for_pyannote() -> None:
    """Снять ограничение weights_only при загрузке чекпоинтов pyannote.

    С torch 2.6 torch.load по умолчанию грузит чекпоинты в безопасном режиме
    (weights_only=True). Чекпоинты pyannote легитимно содержат произвольные
    классы (Specifications, Resolution, ...), поэтому безопасный режим их
    отклоняет. Источник моделей официальный (HuggingFace), риск исполнения
    чужого кода отсутствует — для lightning_fabric._load возвращаем поведение
    до 2.6.

    Порядок обязателен: pyannote.audio.core.model биндит pl_load из
    lightning_fabric при своём импорте, поэтому подменять нужно до импорта
    pyannote. Заодно вносятся в allowlist все классы pyannote.audio.core.task
    — на случай загрузки через другие пути.
    """
    import torch
    from torch.torch_version import TorchVersion

    import lightning_fabric.utilities.cloud_io as cloud_io

    original_load = cloud_io._load

    def _load_trusted(path: str, *args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original_load(path, *args, **kwargs)

    cloud_io._load = _load_trusted

    import pyannote.audio.core.task as task_module

    extra = [obj for obj in vars(task_module).values() if isinstance(obj, type)]
    torch.serialization.add_safe_globals([TorchVersion, *extra])

# ---------------------------------------------------------------------------
# Кэш результатов этапов
#
# Ключ — срез SHA-1 от абсолютного пути входа и параметров этапа: любое
# изменение входа или конфигурации инвалидирует кэш автоматически.
# ---------------------------------------------------------------------------


def cache_key(audio_path: str, *parts: Any) -> str:
    """Построить ключ кэша из пути к аудио и параметров запуска."""
    digest = hashlib.sha1()
    digest.update(os.path.abspath(audio_path).encode("utf-8"))
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()[:16]


def load_cache(cache_dir: str, key: str, stage: str) -> list[dict] | None:
    """Прочитать сохранённый результат этапа; None, если кэша нет."""
    path = os.path.join(cache_dir, f"{key}.{stage}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_cache(cache_dir: str, key: str, stage: str, data: list[dict]) -> None:
    """Сохранить результат этапа; каталог создаётся при необходимости."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.{stage}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Этап 1: распознавание речи
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    """True, если torch видит CUDA-устройство. Импорт внутри функции,
    чтобы не тянуть torch ради одной проверки."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def transcribe_whisper(
    path: str,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
    cache_dir: str | None,
    key: str,
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
    cached = None if cache_dir is None else load_cache(cache_dir, key, "whisper")
    if cached is not None:
        ok_line(f"Whisper: кэш, {len(cached)} слов")
        return cached, time.perf_counter() - started, None

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

    if cache_dir is not None:
        save_cache(cache_dir, key, "whisper", words)
    return words, elapsed, info.duration

# ---------------------------------------------------------------------------
# Этап 2: диаризация
# ---------------------------------------------------------------------------


def load_audio_mono16k(path: str) -> Any:
    """Загрузить аудио как (waveform, 16000 Гц) моно для pyannote.

    pyannote 3.x декодирует через soundfile/libsndfile: wav/mp3/flac читает,
    но видеоконтейнеры (mp4/m4a с AAC) — нет, хотя faster-whisper их ест
    (он декодирует через PyAV). Поэтому при ошибке декодирования
    перекодируем файл во временный WAV через ffmpeg, если он есть в PATH.
    """
    from pyannote.audio import Audio

    audio = Audio(sample_rate=16000, mono="downmix")
    try:
        return audio(path)
    except Exception:
        pass

    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Файл {path!r} не читается декодером pyannote, а ffmpeg в PATH не найден"
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", tmp.name],
            check=True,
            capture_output=True,
        )
        return audio(tmp.name)
    finally:
        os.unlink(tmp.name)


def run_pipeline_with_progress(
    pipeline: Any,
    path: str,
    speaker_count_kwargs: dict[str, int],
) -> Any:
    """Выполнить диаризацию по аудио, один раз загруженному в память.

    На этапе embeddings pyannote запрашивает тысячи перекрывающихся окон.
    Если передать путь к сжатому файлу, каждое окно читается из сжатого файла
    отдельно; на длинных записях это может занимать намного больше времени,
    чем сама нейросетевая обработка. Поэтому заранее декодируем файл
    в моно 16 кГц (load_audio_mono16k сама разбирается с видеоконтейнерами).
    speaker_count_kwargs — min_speakers/max_speakers: главный рычаг против
    over-clustering, когда pyannote дробит одного спикера на кластеры.
    """
    waveform, sample_rate = load_audio_mono16k(path)
    audio_file = {
        "uri": os.path.basename(path),
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError:
        # Совместимость с версиями pyannote без ProgressHook.
        return pipeline(audio_file, **speaker_count_kwargs)

    with ProgressHook() as hook:
        return pipeline(audio_file, hook=hook, **speaker_count_kwargs)


def torch_device(index: int) -> Any:
    import torch

    return torch.device(f"cuda:{index}")


def diarize(
    path: str,
    hf_token: str | None,
    device: str,
    cache_dir: str | None,
    key: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[list[dict], float]:
    """Определить спикеров и вернуть (интервалы, длительность_этапа).

    Каждый интервал — dict {start, end, speaker} с непересекающимися
    отрезками одного спикера. min_speakers/max_speakers передаются в
    пайплайн как есть (None = авто): на длинных записях pyannote склонен
    дробить спикеров на лишние кластеры, и ограничение числа спикеров —
    самое надёжное лекарство. Патчи совместимости применяются здесь,
    непосредственно перед импортом pyannote (см. требования к порядку).
    """
    started = time.perf_counter()
    cached = None if cache_dir is None else load_cache(cache_dir, key, "diarization")
    if cached is not None:
        ok_line(f"Диаризация: кэш, {len(cached)} фрагментов")
        return cached, time.perf_counter() - started

    if not hf_token:
        sys.exit(
            f"{red('Ошибка:')} не задан токен HuggingFace "
            "(переменная HF_TOKEN или аргумент --hf-token)"
        )

    register_ffmpeg_dll_dirs()
    patch_torchaudio_for_pyannote3()
    patch_torch_load_for_pyannote()

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        sys.exit(
            f"{red('Ошибка:')} не удалось импортировать pyannote.audio: {exc}\n"
            f"        Проверьте установку связки версий из README.md"
        )

    stage_header(2, "Диаризация", "pyannote/speaker-diarization-3.1")
    print(f"        {dim('Загрузка моделей (первый запуск: ~2.5 ГБ, дальше из кэша)...')}")
    # На этапе 1 прогресс-бары huggingface_hub выключены глобально
    # (см. раздел окружения), а здесь долгая сетевая загрузка — включаем
    # обратно, чтобы первый запуск не выглядел зависшим. enable_progress_bars
    # уступает приоритет переменной окружения, поэтому снимаем её.
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    try:
        from huggingface_hub.utils import enable_progress_bars

        enable_progress_bars()
    except Exception:
        pass

    # pyannote >= 4.0 принимает token, 3.x — use_auth_token. Выбираем
    # аргумент по сигнатуре, чтобы TypeError из самой загрузки модели не
    # перехватывался и не запускал её повторно с другими параметрами.
    auth_params = inspect.signature(Pipeline.from_pretrained).parameters
    auth_kwargs = (
        {"token": hf_token}
        if "token" in auth_params
        else {"use_auth_token": hf_token}
    )
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **auth_kwargs,
        )
    except Exception as exc:
        sys.exit(
            f"{red('Ошибка загрузки модели диаризации:')} {type(exc).__name__}: {exc}\n"
            f"        Типичные причины: токен не задан/неверен, не приняты условия\n"
            f"        gated-моделей pyannote на huggingface.co, конфликт версий\n"
            f"        (см. README.md, раздел «Устранение проблем»)"
        )
    if device == "cuda":
        try:
            pipeline.to(torch_device(0))
        except Exception as exc:
            sys.exit(
                f"{red('Ошибка переноса пайплайна на GPU:')} {type(exc).__name__}: {exc}\n"
                f"        Похоже на нехватку VRAM — попробуйте --device cpu"
            )

    print(f"        {dim('Этапы пайплайна:')}")
    speaker_count_kwargs: dict[str, int] = {}
    if min_speakers is not None:
        speaker_count_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        speaker_count_kwargs["max_speakers"] = max_speakers
    try:
        diarization = run_pipeline_with_progress(pipeline, path, speaker_count_kwargs)
    except Exception as exc:
        sys.exit(
            f"{red('Ошибка во время диаризации:')} {type(exc).__name__}: {exc}\n"
            f"        Типичные причины: формат не поддержан декодером pyannote\n"
            f"        (нужен ffmpeg в PATH), ошибка torchcodec (нужен ffmpeg с DLL\n"
            f"        или pyannote.audio<4), нехватка VRAM. Если спикеров слишком\n"
            f"        много/мало — уточните --min-speakers/--max-speakers (см. README.md)"
        )
    turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    elapsed = time.perf_counter() - started
    ok_line(f"{len(turns)} фрагментов · {fmt_mmss(elapsed)}")

    if cache_dir is not None:
        save_cache(cache_dir, key, "diarization", turns)
    return turns, elapsed

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


def fmt_timestamp(seconds: float) -> str:
    """Форматировать секунды как hh:mm:ss — таймкоды в выходном файле."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

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
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
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

    # HF_HOME читается huggingface_hub при импорте, поэтому задаём до
    # импорта любых зависимостей, использующих его (Whisper, pyannote).
    os.environ.setdefault(
        "HF_HOME", os.path.abspath(os.path.join(args.cache_dir, "hf"))
    )

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

    cache_dir = None if args.no_cache else args.cache_dir
    key = cache_key(
        args.audio, args.model, args.language, args.compute_type, args.device,
        args.min_speakers, args.max_speakers,
    )

    started = time.perf_counter()
    words, time_whisper, duration = transcribe_whisper(
        args.audio, args.model, args.language, args.device,
        args.compute_type, cache_dir, key,
    )
    turns, time_diarization = diarize(
        args.audio, args.hf_token, args.device, cache_dir, key,
        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
    )
    time_assign = assign_speakers(words, turns)

    lines = group_into_lines(words)
    total = time.perf_counter() - started

    if not lines:
        sys.exit(f"{red('Ничего не распознано')} — проверьте качество аудио")

    # Нумерация спикеров по убыванию говорливости: SPEAKER_00 — самый
    # активный; человеческие имена подставляются поиском-заменой по файлу.
    counts: dict[str, int] = {}
    for line in lines:
        counts[line["speaker"]] = counts.get(line["speaker"], 0) + 1
    order = sorted(counts, key=counts.get, reverse=True)
    rename = {sp: f"SPEAKER_{i:02d}" for i, sp in enumerate(order)}

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