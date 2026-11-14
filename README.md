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

## Содержание

- [Требования](#требования)
- [Установка](#установка)
- [Токен HuggingFace](#токен-huggingface)
- [Запуск](#запуск)
- [Сценарии](#сценарии)
- [Пакетная обработка (transcribe_command.sh)](#пакетная-обработка-transcribe_commandsh)
- [Как это работает](#как-это-работает)
- [Структура проекта](#структура-проекта)
- [Результат](#результат)
- [Устранение проблем](#устранение-проблем)
- [Скорость](#скорость)

## Требования

- **Python 3.10–3.13**
- Windows / Linux / macOS
- **GPU NVIDIA с 6+ ГБ VRAM** настоятельно рекомендуется: на CPU тот же час аудио займёт ~час даже с `medium`
- ffmpeg — только для pyannote.audio 4.x (со связкой из [«Установки»](#установка) **не нужен**)

## Установка

Ставьте именно эту проверенную связку версий — отклонения чреваты конфликтами (см. [«Устранение проблем»](#устранение-проблем)):

```bash
pip install faster-whisper "pyannote.audio==3.4.0" tqdm  # пин 3.4.0: 4.x требует ffmpeg/torchcodec
pip install "av<19"  # PyAV 19 ломает faster-whisper 1.2.1
pip install "huggingface_hub<1.0"  # hub 1.x убрал use_auth_token, нужный pyannote 3.x
# GPU NVIDIA; без GPU пропустите строку ниже:
pip install --force-reinstall "torch==2.7.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
```

Проверка CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
```

Если `False` — torch встал CPU-сборкой, переустановите последней строкой выше.

## Токен HuggingFace

Диаризация использует закрытые (gated) модели pyannote:

1. Регистрация: https://huggingface.co
2. Токен: https://huggingface.co/settings/tokens → New token → тип **Read**
3. Принять условия на трёх страницах (кнопка _Agree and access repository_):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1

```bash
echo "HF_TOKEN=hf_xxxxxxxx" > .env   # .env в .gitignore, в git не попадает
```

Без токена или без принятых условий будет `GatedRepoError: 403 Forbidden` — скрипт подскажет, какую именно модель одобрить. Разовая настройка, дальше всё в кэше. `transcribe_command.sh` подхватывает `.env` сам (см. [«Пакетная обработка»](#пакетная-обработка-transcribe_commandsh)).

## Запуск

```bash
python transcribe.py meeting.mp3 --language ru --device cuda --compute-type float16
```

Устройство по умолчанию подбирается автоматически (CUDA, если доступна), поэтому
`python transcribe.py meeting.mp3 --language ru` работает и без флагов.

| Параметр         | По умолчанию          | Описание                                         |
| ---------------- | --------------------- | ------------------------------------------------ |
| `audio`          | —                     | путь к аудиофайлу (mp3, wav, m4a, ogg...)        |
| `--out`          | `transcript.txt`      | куда сохранить результат                         |
| `--model`        | `large-v3`            | tiny / base / small / medium / large-v3          |
| `--language`     | авто                  | код языка: `ru`, `en`...                         |
| `--device`       | cuda, если доступен   | `cuda` или `cpu`; недоступная CUDA — авто-переключение на `cpu` |
| `--compute-type` | `float16`             | `float16` (GPU) / `int8`; на CPU принудительно `int8` |
| `--min-speakers` | авто                  | минимальное число спикеров (см. ниже)            |
| `--max-speakers` | авто                  | максимальное число спикеров (см. ниже)           |
| `--cache-dir`    | `cache`               | модели и промежуточные результаты                |
| `--no-cache`     | —                     | пересчитать всё заново                           |
| `--hf-token`     | переменная `HF_TOKEN` | токен HuggingFace (удобнее задать через `.env`)  |

Про `--min-speakers`/`--max-speakers`: pyannote на длинных записях склонен дробить одного спикера на несколько кластеров (over-clustering) — типичный симптом: много «спикеров», суммарно говорящих по секунде-две. Если число участников известно, уточните его:

```bash
python transcribe.py meeting.mp3 --language ru --min-speakers 2 --max-speakers 2
```

## Сценарии

```bash
# Есть GPU — быстро и точно
python transcribe.py meeting.mp3 --language ru --device cuda --compute-type float16

# Нет GPU — medium на CPU разумный компромисс
python transcribe.py meeting.mp3 --language ru --model medium --device cpu

# Слабая GPU (мало VRAM)
python transcribe.py meeting.mp3 --language ru --device cuda --compute-type int8
```

## Пакетная обработка (transcribe_command.sh)

Обрабатывает все `.mp3`/`.mp4` в папке проекта: из MP4 сначала извлекается аудио через ffmpeg, транскрипт каждого файла сохраняется рядом с ним: `meeting.mp3` → `meeting.txt`. Токен берётся из `.env` (см. [«Токен HuggingFace»](#токен-huggingface)), число спикеров задаётся в первых строках скрипта (`MIN_SPEAKERS`/`MAX_SPEAKERS`, пустое значение = автоопределение).

```bash
bash transcribe_command.sh
```

## Как это работает

1. **Распознавание** (faster-whisper): аудио → слова с таймкодами. Кэшируется в `cache/*.whisper.json` — повторные запуски мгновенные.
2. **Диаризация** (pyannote): аудио → фрагменты с метками спикеров. Кэшируется в `cache/*.diarization.json`.
3. **Привязка**: каждому слову назначается спикер по максимальному временному пересечению, слова склеиваются в реплики.

Все модели и результаты живут в `cache/` рядом со скриптом. Полный сброс: `rm -rf cache`.

## Структура проекта

```
.
├── transcribe.py            # точка входа: python transcribe.py <args>
├── transcribe/              # код конвейера
│   ├── cli.py               # аргументы и оркестрация этапов
│   ├── recognizer.py        # faster-whisper: распознавание + кэш
│   ├── diarizer.py          # pyannote: диаризация + патчи совместимости + кэш
│   ├── aligner.py           # привязка слов к спикерам, склейка реплик
│   ├── cache.py             # кэш результатов этапов (StageCache)
│   ├── config.py            # окружение, CUDA, HF_TOKEN
│   └── output.py            # форматирование консоли и итогового транскрипта
├── transcribe_command.sh    # пакетная обработка mp3/mp4 в папке проекта
└── cache/                   # модели и результаты этапов (в .gitignore)
```

## Результат

Имена `SPEAKER_00`, `SPEAKER_01` — нумерация по убыванию говорливости (№0 говорил больше всех). Настоящие имена подставляются поиском-заменой по файлу. Слова, которые не удалось привязать к интервалу диаризации, помечены `UNKNOWN`.

## Устранение проблем

Современные версии пакетов конфликтуют между собой — **самое надёжное лекарство: поставить связку версий из [«Установки»](#установка)**.

| Ошибка / симптом | Причина | Решение |
| --- | --- | --- |
| `GatedRepoError: 403 Forbidden` | нет токена или не приняты условия gated-модели | токен (тип Read) в `.env` + открыть модель из ошибки на hf.co → Agree |
| Очень долго | CPU + large-v3 | `--device cuda` или `--model medium` |
| Слишком много спикеров | pyannote дробит кластеры (over-clustering) | уточнить `--min-speakers`/`--max-speakers`, если число участников известно |
| `torch.cuda.is_available()` → `False` | CPU-сборка torch | переустановить из cu128-индекса (см. «Установка») |
| `Could not load libtorchcodec` | установлена pyannote.audio 4.x | `pip install "pyannote.audio==3.4.0"`, либо ffmpeg **full-shared** с gyan.dev |
| `open() got an unexpected keyword argument 'metadata_errors'` | PyAV 19.x несовместим с faster-whisper 1.2.1 | `pip install "av<19"` |
| `hf_hub_download() got an unexpected keyword argument 'use_auth_token'` | huggingface_hub 1.x | `pip install "huggingface_hub<1.0"` |
| `Pipeline.from_pretrained() got an unexpected keyword argument 'token'` | конфликт версий pyannote/hub (скрипт сам подбирает аргумент) | надёжнее — связка версий из «Установки» |
| `module 'torchaudio' has no attribute 'AudioMetaData'` / `list_audio_backends` | torchaudio слишком новый для pyannote 3.x | `pip install --force-reinstall "torchaudio==2.7.1" "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128` |
| `Weights only load failed: Unsupported global pyannote...` | torch ≥ 2.6 грузит чекпоинты в безопасном режиме | обрабатывается патчем в скрипте; при сбое — связка версий из «Установки» |

## Скорость

- RTX 4070 Ti: ~3 мин распознавания + ~10 мин диаризации на час аудио (замер: 43 мин записи — 2 и 7 мин)
- Современный CPU с `medium`: ~30–60 мин на час аудио
