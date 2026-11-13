# Transcribe — транскрипция аудио с разделением по спикерам

Скрипт на Python: распознаёт речь (Whisper) и определяет, кто когда говорил (диаризация pyannote). На выходе — текстовый файл с репликами, разбитыми по спикерам и с таймкодами.

```
[00:03:15] SPEAKER_00: Добрый день, давайте начнём встречу.
[00:03:22] SPEAKER_01: Здравствуйте, у меня пара вопросов по бюджету.
```

## Требования

- **Python 3.10–3.13**
- Windows / Linux / macOS
- **GPU NVIDIA с 6+ ГБ VRAM** настоятельно рекомендуется: час аудио на GPU ~2 минуты против ~часа на CPU с large-v3
- ffmpeg — только для pyannote.audio 4.x (со связкой ниже **не нужен**)

## Установка

Рабочая проверенная связка версий (важно ставить именно её, см. «Устранение проблем»):

```bash
pip install faster-whisper "pyannote.audio==3.4.0" tqdm   # 3.4.0: версия 4.x требует ffmpeg/torchcodec — см. «Устранение проблем»
pip install "av<19"                      # PyAV 19 ломает faster-whisper 1.2.1
pip install "huggingface_hub<1.0"        # hub 1.x убрал use_auth_token, нужный pyannote 3.x
pip install --force-reinstall "torch==2.7.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128   # для GPU NVIDIA
```

Проверка CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
```

Если `False` — torch встал CPU-сборкой, переустановите командой выше. Если GPU нет — просто пропустите третью строку и запускайте с `--model medium --device cpu`.

## Токен HuggingFace

Диаризация использует закрытые (gated) модели pyannote:

1. Регистрация: https://huggingface.co
2. Токен: https://huggingface.co/settings/tokens → New token → тип **Read**
3. Принять условия на трёх страницах (кнопка _Agree and access repository_):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1

   Без этого будет `GatedRepoError: 403 Forbidden` — скрипт сообщит, какую именно модель нужно одобрить. Разовая настройка, дальше всё в кэше.

4. Передать токен — удобнее всего через `.env` в папке проекта:

```bash
echo "HF_TOKEN=hf_xxxxxxxx" > .env   # .env в .gitignore, в git не попадает
```

   Скрипт `transcribe_command.sh` подхватывает `.env` автоматически; `transcribe.py`
   читает `HF_TOKEN` из окружения (переменная оболочки или `.env`, подгруженный
   скриптом). Без токена будет `GatedRepoError: 403 Forbidden` — скрипт сообщит,
   какую именно модель нужно одобрить. Разовая настройка, дальше всё в кэше.

## Пакетная обработка (transcribe_command.sh)

Обрабатывает все `.mp3`/`.mp4` в текущей папке: из MP4 сначала извлекается
аудио через ffmpeg, затем для каждого файла запускается транскрипция.
Токен берётся из `.env` (см. выше), число спикеров — в первых строках скрипта
(`MIN_SPEAKERS`/`MAX_SPEAKERS`, пустое значение = автоопределение).
Транскрипт каждого файла сохраняется рядом с ним: `meeting.mp3` → `meeting.txt`.
Скрипт закоммичен в репозиторий и безопасен для публикации — секретов в нём нет.

```bash
bash transcribe_command.sh
```

## Запуск

```bash
python transcribe.py audio.mp3 --out transcript.txt --language ru --device cuda --compute-type float16
```

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

Про `--min-speakers`/`--max-speakers`: pyannote на длинных записях склонен дробить
одного спикера на несколько кластеров (over-clustering) — типичный симптом: много
«спикеров», суммарно говорящих по секунде-две. Если число участников известно,
уточните его:

```bash
python transcribe.py meeting.mp3 --language ru --min-speakers 2 --max-speakers 2
```

Сценарии:

```bash
# Есть GPU — быстро и точно
python transcribe.py meeting.mp3 --language ru --device cuda --compute-type float16

# Нет GPU — medium на CPU разумный компромисс
python transcribe.py meeting.mp3 --language ru --model medium --device cpu

# Слабая GPU (мало VRAM)
python transcribe.py meeting.mp3 --language ru --device cuda --compute-type int8
```

## Как это работает

1. **Распознавание** (faster-whisper): аудио → слова с таймкодами. Кэшируется в `cache/*.whisper.json` — повторные запуски мгновенные.
2. **Диаризация** (pyannote): аудио → фрагменты с метками спикеров. Кэшируется в `cache/*.diarization.json`.
3. **Привязка**: каждому слову назначается спикер по максимальному временному пересечению, слова склеиваются в реплики.

Все модели и результаты живут в `cache/` рядом со скриптом. Полный сброс: `rm -rf cache`.

## Результат

Имена `SPEAKER_00`, `SPEAKER_01` — нумерация по убыванию говорливости (№0 говорил больше всех). Настоящие имена подставляются поиском-заменой по файлу.

## Устранение проблем

Современные версии пакетов конфликтуют между собой — **самое надёжное лекарство: поставить связку версий из раздела «Установка»**.

| Ошибка                                                                         | Причина                                                     | Решение                                                                                                               |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `open() got an unexpected keyword argument 'metadata_errors'`                  | PyAV 19.x несовместим с faster-whisper 1.2.1                | `pip install "av<19"`                                                                                                 |
| `GatedRepoError: 403 Forbidden`                                                | не приняты условия модели                                   | открыть модель из ошибки на hf.co → Agree                                                                             |
| `torch.cuda.is_available()` → `False`                                          | CPU-сборка torch                                            | переустановить из cu128-индекса (см. выше)                                                                            |
| `Pipeline.from_pretrained() got an unexpected keyword argument 'token'`        | pyannote 3.x (скрипт сам переключается на `use_auth_token`) | ничего, в скрипте есть фолбэк                                                                                         |
| `hf_hub_download() got an unexpected keyword argument 'use_auth_token'`        | huggingface_hub 1.x                                         | `pip install "huggingface_hub<1.0"`                                                                                   |
| `module 'torchaudio' has no attribute 'AudioMetaData'` / `list_audio_backends` | torchaudio слишком новый для pyannote 3.x                   | `pip install --force-reinstall "torchaudio==2.7.1" "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128` |
| `Weights only load failed: Unsupported global pyannote...`                     | torch ≥ 2.6 грузит чекпоинты в безопасном режиме            | есть в скрипте (патч), либо связка версий из «Установки»                                                              |
| `Could not load libtorchcodec`                                                 | pyannote 4.x + torchcodec на Windows                        | перейти на `pyannote.audio<4` (связка из «Установки»), либо ffmpeg **full-shared** с gyan.dev                         |
| Очень долго                                                                    | CPU + large-v3                                              | `--device cuda` или `--model medium`                                                                                  |
| Слишком много спикеров                                                         | pyannote дробит кластеры (over-clustering)                  | уточнить `--min-speakers`/`--max-speakers`, если число участников известно                                            |

## Скорость

- RTX 4070 Ti: ~2 мин распознавания + ~7 мин диаризации на 43 мин аудио
- Современный CPU с `medium`: ~30–60 мин на час аудио
