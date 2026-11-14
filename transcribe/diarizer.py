"""Этап 2: диаризация (pyannote.audio) с кэшем результатов.

Включает патчи совместимости платформ и версий: каждый патч здесь —
обходной путь для конкретного конфликта версий; если зависимости
обновились и конфликт исчез, патч должен стать безопасной no-op.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from typing import Any

from transcribe.cache import StageCache
from transcribe.output import dim, fmt_mmss, ok_line, red, stage_header

# Модель диаризации (gated: нужен HF_TOKEN и принятые условия на hf.co).
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Кандидаты на расположение bin-директории ffmpeg. Используется только на
# Windows и только для связки pyannote.audio 4.x + torchcodec; pyannote 3.x
# декодирует через soundfile и ffmpeg не требует.
FFMPEG_BIN_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files\ffmpeg\bin",
    r"C:\ffmpeg\bin",
)

# Хендлы os.add_dll_directory: без ссылки на них каталог тут же выводится
# из поиска DLL сборщиком мусора, и патч для torchcodec превращается в no-op.
_dll_dir_handles: list[Any] = []


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
            _dll_dir_handles.append(os.add_dll_directory(path))


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
    cache: StageCache,
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
    cached = cache.load("diarization")
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

    stage_header(2, "Диаризация", DIARIZATION_MODEL)
    print(f"        {dim('Загрузка моделей (первый запуск: ~2.5 ГБ, дальше из кэша)...')}")
    # На этапе 1 прогресс-бары huggingface_hub выключены глобально
    # (см. transcribe/config.py), а здесь долгая сетевая загрузка — включаем
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
            DIARIZATION_MODEL,
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

    cache.save("diarization", turns)
    return turns, elapsed
