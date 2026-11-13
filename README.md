<div align="center">
  <img src="./project_icon.png" width="140" alt="Transcribe — иконка: звуковая волна превращается в реплики двух спикеров">
  <h1>Transcribe</h1>
  <p>Транскрипция аудио с разделением по спикерам: распознаёт речь (faster-whisper)<br>
  и определяет, кто когда говорил (диаризация pyannote.audio).<br>
  На выходе — текстовый файл с репликами и таймкодами.</p>
  <img src="https://img.shields.io/static/v1?label=python&message=3.10-3.13&color=blue" alt="Python 3.10–3.13">
  <img src="https://img.shields.io/static/v1?label=platform&message=Windows%20%7C%20Linux%20%7C%20macOS&color=lightgrey" alt="Платформы: Windows, Linux, macOS">
</div>

Пример фрагмента результата:

```text
[00:03:15] SPEAKER_00: Добрый день, давайте начнём встречу.
[00:03:22] SPEAKER_01: Здравствуйте, у меня пара вопросов по бюджету.
```

## Быстрый старт

**Шаг 1. Установите зависимости.** Ставьте именно эту проверенную связку версий — отклонения чреваты конфликтами (см. «Устранение проблем»). Нужны Python 3.10–3.13, Windows / Linux / macOS.

```bash
pip install faster-whisper "pyannote.audio==3.4.0" tqdm   # 3.4.0: версия 4.x требует ffmpeg/torchcodec — см. «Устранение проблем»
pip install "av<19"                      # PyAV 19 ломает faster-whisper 1.2.1
pip install "huggingface_hub<1.0"        # hub 1.x убрал use_auth_token, нужный pyannote 3.x
pip install --force-reinstall "torch==2.7.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128   # для GPU NVIDIA
```

Последняя строка — для GPU NVIDIA. **GPU с 6+ ГБ VRAM настоятельно рекомендуется**: на CPU тот же час аудио займёт ~час даже с `medium`. Без GPU строку пропустите.

Проверка CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
```

Если `False` — torch встал CPU-сборкой, переустановите последней строкой выше.

**Шаг 2. Создайте токен HuggingFace.** Диаризация использует закрытые (gated) модели pyannote:

1. Регистрация: https://huggingface.co
2. Токен: https://huggingface.co/settings/tokens → New token → тип **Read**
3. Принять условия на трёх страницах (кнопка _Agree and access repository_):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1

```bash
echo "HF_TOKEN=hf_xxxxxxxx" > .env   # .env в .gitignore, в git не попадает
```

Без токена или без принятых условий будет `GatedRepoError: 403 Forbidden` — скрипт подскажет, какую именно модель одобрить. Разовая настройка, дальше всё в кэше.

**Шаг 3. Запустите:**

```bash
python transcribe.py meeting.mp3 --language ru
```

Устройство подбирается автоматически (CUDA, если доступна). Готово — результат в `transcript.txt`:

```
[00:03:15] SPEAKER_00: Добрый день, давайте начнём встречу.
```

Имена `SPEAKER_00`, `SPEAKER_01` — нумерация по убыванию говорливости (№0 говорил больше всех). Настоящие имена подставляются поиском-заменой по файлу.

## Параметры

| Параметр         | По умолчанию        | Описание                                         |
| ---------------- | ------------------- | ------------------------------------------------ |
| `audio`          | —                   | путь к аудиофайлу (mp3, wav, m4a, ogg...)        |
| `--out`          | `transcript.txt`    | куда сохранить результат                         |
| `--model`        | `large-v3`          | tiny / base / small / medium / large-v3          |
| `--language`     | авто                | код языка: `ru`, `en`...                         |
| `--device`       | cuda, если доступен | `cuda` или `cpu`                                 |
| `--compute-type` | `float16`           | `float16` (GPU) / `int8` (CPU или экономия VRAM) |
| `--min-speakers` | авто                | минимальное число спикеров (см. ниже)            |
| `--max-speakers` | авто                | максимальное число спикеров (см. ниже)           |
| `--cache-dir`    | `cache`             | модели и промежуточные результаты                |
| `--no-cache`     | —                   | пересчитать всё заново                           |

Про `--min-speakers`/`--max-speakers`: pyannote на длинных записях склонен дробить одного спикера на несколько кластеров (over-clustering) — типичный симптом: много «спикеров», суммарно говорящих по секунде-две. Если число участников известно, уточните его:

```bash
python transcribe.py meeting.mp3 --language ru --min-speakers 2 --max-speakers 2
```

Типовые сценарии:

```bash
# Есть GPU — быстро и точно
python transcribe.py meeting.mp3 --language ru --device cuda --compute-type float16

# Нет GPU — medium на CPU разумный компромисс
python transcribe.py meeting.mp3 --language ru --model medium --device cpu

# Слабая GPU (мало VRAM)
python transcribe.py meeting.mp3 --language ru --device cuda --compute-type int8
```

## Пакетная обработка (transcribe_command.sh)

Обрабатывает все `.mp3`/`.mp4` в папке проекта: из MP4 сначала извлекается аудио через ffmpeg, транскрипт каждого файла сохраняется рядом с ним: `meeting.mp3` → `meeting.txt`. Токен берётся из `.env` (см. шаг 2), число спикеров задаётся в первых строках скрипта (`MIN_SPEAKERS`/`MAX_SPEAKERS`, пустое значение = автоопределение).

```bash
bash transcribe_command.sh
```

## Как это работает

1. **Распознавание** (faster-whisper): аудио → слова с таймкодами. Кэшируется в `cache/*.whisper.json` — повторные запуски мгновенные.
2. **Диаризация** (pyannote): аудио → фрагменты с метками спикеров. Кэшируется в `cache/*.diarization.json`.
3. **Привязка**: каждому слову назначается спикер по максимальному временному пересечению, слова склеиваются в реплики.

Все модели и результаты живут в `cache/` рядом со скриптом. Полный сброс: `rm -rf cache`.

## Устранение проблем

Современные версии пакетов конфликтуют между собой — **самое надёжное лекарство: поставить связку версий из «Быстрого старта»**.

| Ошибка / симптом                                                               | Причина                                                     | Решение                                                                                                               |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `GatedRepoError: 403 Forbidden`                                                | нет токена или не приняты условия gated-модели              | токен (тип Read) в `.env` + открыть модель из ошибки на hf.co → Agree                                                 |
| Очень долго                                                                    | CPU + large-v3                                              | `--device cuda` или `--model medium`                                                                                  |
| Слишком много спикеров                                                         | pyannote дробит кластеры (over-clustering)                  | уточнить `--min-speakers`/`--max-speakers`, если число участников известно                                            |
| `torch.cuda.is_available()` → `False`                                          | CPU-сборка torch                                            | переустановить из cu128-индекса (см. «Быстрый старт»)                                                                 |
| `Could not load libtorchcodec`                                                 | установлена pyannote.audio 4.x                              | `pip install "pyannote.audio==3.4.0"`, либо ffmpeg **full-shared** с gyan.dev                                         |
| `open() got an unexpected keyword argument 'metadata_errors'`                  | PyAV 19.x несовместим с faster-whisper 1.2.1                | `pip install "av<19"`                                                                                                 |
| `hf_hub_download() got an unexpected keyword argument 'use_auth_token'`        | huggingface_hub 1.x                                         | `pip install "huggingface_hub<1.0"`                                                                                   |
| `Pipeline.from_pretrained() got an unexpected keyword argument 'token'`        | конфликт версий pyannote/hub (скрипт сам подбирает аргумент) | надёжнее — связка версий из «Быстрого старта»                                                                        |
| `module 'torchaudio' has no attribute 'AudioMetaData'` / `list_audio_backends` | torchaudio слишком новый для pyannote 3.x                   | `pip install --force-reinstall "torchaudio==2.7.1" "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128` |
| `Weights only load failed: Unsupported global pyannote...`                     | torch ≥ 2.6 грузит чекпоинты в безопасном режиме            | обрабатывается патчем в скрипте; при сбое — связка версий из «Быстрого старта»                                        |

## Скорость

- RTX 4070 Ti: ~3 мин распознавания + ~10 мин диаризации на час аудио (замер: 43 мин записи — 2 и 7 мин)
- Современный CPU с `medium`: ~30–60 мин на час аудио
