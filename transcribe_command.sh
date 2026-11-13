# Токен HuggingFace нужен для диаризации и хранится в .env (файл в
# .gitignore, в репозиторий не попадает). Скрипт подхватывает его сам:
# просто положите рядом .env вида  HF_TOKEN=hf_xxxxxxxx
# Если .env нет — используется HF_TOKEN из окружения оболочки.
#
# Токен: https://huggingface.co/settings/tokens (тип Read);
# также нужно принять условия gated-моделей pyannote (см. README.md).

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source ./.env
    set +a
fi

if [ -z "$HF_TOKEN" ]; then
    echo "!! ВНИМАНИЕ: HF_TOKEN не задан — диаризация завершится с ошибкой!"
    echo "   Создайте .env с токеном или задайте вручную: export HF_TOKEN=hf_xxxxxxxx"
fi

# python transcribe.py 1102.MP3 --out transcript.txt --language ru --device cuda --compute-type float16

# Включаем игнорирование регистра при поиске файлов (.mp3, .MP3, .mp4, .MP4)
shopt -s nullglob nocaseglob

for f in *.mp3 *.mp4; do 
    clear
    echo "=== $f ==="

    # Проверяем, является ли файл видео (приводим имя к нижнему регистру)
    if [[ "${f,,}" == *.mp4 ]]; then
        # Если это MP4, сначала извлекаем аудио
        temp_audio="temp_${f%.*}.mp3"
        echo ">> Извлечение аудио из видео..."
        ffmpeg -y -i "$f" -vn -acodec libmp3lame -q:a 2 "$temp_audio" -loglevel error
        
        if [ -f "$temp_audio" ]; then
            # Запускаем извлечение аудио
            python transcribe.py "$temp_audio" --out /dev/null --language ru --device cuda --compute-type float16
            rm -f "$temp_audio" # Удаляем временный файл
        else
            echo "!! Ошибка: не удалось извлечь аудио из $f"
        fi
    else
        # Работа с изначальным mp3
        python transcribe.py "$f" --out /dev/null --language ru --device cuda --compute-type float16
    fi
    
done