#!/usr/bin/env bash
# Токен HuggingFace нужен для диаризации и хранится в .env (файл в
# .gitignore, в репозиторий не попадает). Скрипт подхватывает его сам:
# просто положите рядом .env вида  HF_TOKEN=hf_xxxxxxxx
# Если .env нет — используется HF_TOKEN из окружения оболочки.
#
# Токен: https://huggingface.co/settings/tokens (тип Read);
# также нужно принять условия gated-моделей pyannote (см. README.md).

# Падать при первой непредвиденной ошибке (нехватка прав, отсутствующие
# команды), а не продолжать полуразобранный прогон. Ожидаемые ошибки —
# ffmpeg и transcribe.py — перехватываются явно через if ниже, поэтому
# обработка файлов продолжается, а счётчики ok/failed ведутся как раньше.
set -e

# Работаем в папке скрипта: отсюда берутся .env и папки input/output,
# независимо от того, откуда запущен скрипт.
cd "$(dirname "$0")" || exit 1

if [ -f .env ]; then
    set -a
    # sed срезает \r: файл, отредактированный в Блокноте с CRLF, иначе
    # дал бы переменным значения с «хвостом» \r и сломал бы токен.
    # shellcheck disable=SC1091
    source <(sed 's/\r$//' ./.env)
    set +a
fi

if [ -z "$HF_TOKEN" ]; then
    echo "!! ВНИМАНИЕ: HF_TOKEN не задан — диаризация завершится с ошибкой!"
    echo "   Создайте .env с токеном или задайте вручную: export HF_TOKEN=hf_xxxxxxxx"
fi

# ═══ НАСТРОЙКИ (изменяйте под свою запись) ═══
MIN_SPEAKERS=2      # минимальное число спикеров (пусто = авто: MIN_SPEAKERS="")
MAX_SPEAKERS=2      # максимальное число спикеров (пусто = авто: MAX_SPEAKERS="")

# Файлы для обработки берутся из input/, готовые транскрипты пишутся
# в output/ под именем исходника: input/meeting.mp3 → output/meeting.txt
INPUT_DIR="input"
OUTPUT_DIR="output"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# Собираем аргументы для transcribe.py из настроек выше
SPEAKER_ARGS=()
if [ -n "$MIN_SPEAKERS" ]; then
    SPEAKER_ARGS+=(--min-speakers "$MIN_SPEAKERS")
fi
if [ -n "$MAX_SPEAKERS" ]; then
    SPEAKER_ARGS+=(--max-speakers "$MAX_SPEAKERS")
fi

# python transcribe.py 1102.MP3 --out transcript.txt --language ru --device cuda --compute-type float16

# Включаем игнорирование регистра при поиске файлов (.mp3, .MP3, .mp4, .MP4)
shopt -s nullglob nocaseglob

# Список файлов фиксируем до цикла: временный mp3 из MP4 не должен
# попасть в обработку повторно.
files=("$INPUT_DIR"/*.mp3 "$INPUT_DIR"/*.mp4)

if [ ${#files[@]} -eq 0 ]; then
    echo "В папке $INPUT_DIR/ нет mp3/mp4 — положите туда файлы для транскрибации."
    exit 0
fi

ok=0       # файлов успешно обработано
failed=0   # файлов с ошибкой

for f in "${files[@]}"; do
    # Очищаем экран только при интерактивном запуске: при перенаправлении
    # вывода в файл escape-последовательности clear засорили бы его.
    if [ -t 1 ]; then
        clear
    fi
    echo "=== $(basename "$f") ==="

    name="$(basename "${f%.*}")"
    out="$OUTPUT_DIR/$name.txt"

    # Проверяем, является ли файл видео (приводим имя к нижнему регистру)
    if [[ "${f,,}" == *.mp4 ]]; then
        # Если это MP4, сначала извлекаем аудио
        temp_audio="$INPUT_DIR/temp_$name.mp3"
        echo ">> Извлечение аудио из видео..."
        # Успех проверяем по коду возврата ffmpeg, а не по факту файла:
        # прерванное извлечение оставляет битый частичный mp3.
        if ffmpeg -y -i "$f" -vn -acodec libmp3lame -q:a 2 "$temp_audio" -loglevel error; then
            # Временный файл удаляем при любом исходе транскрибации.
            if python transcribe.py "$temp_audio" --out "$out" --language ru --device cuda --compute-type float16 "${SPEAKER_ARGS[@]}"; then
                ((++ok))
            else
                echo "!! Ошибка обработки $(basename "$f")"
                ((++failed))
            fi
            rm -f "$temp_audio"
        else
            echo "!! Ошибка: не удалось извлечь аудио из $(basename "$f")"
            rm -f "$temp_audio"
            ((++failed))
            continue
        fi
    else
        # Работа с изначальным mp3
        if python transcribe.py "$f" --out "$out" --language ru --device cuda --compute-type float16 "${SPEAKER_ARGS[@]}"; then
            ((++ok))
        else
            echo "!! Ошибка обработки $(basename "$f")"
            ((++failed))
        fi
    fi
done

echo
echo "════════════════════════════════════════"
echo "  Обработано: $ok · Ошибок: $failed"
echo "  Транскрипты сохранены в папке $OUTPUT_DIR/: <имя>.txt"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
