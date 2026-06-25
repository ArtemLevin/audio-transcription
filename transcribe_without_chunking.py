import argparse  # Удобный разбор аргументов командной строки
import json  # Чтение ответа ffprobe
import shutil  # Проверка наличия ffmpeg / ffprobe
import subprocess  # Запуск внешних команд
from dataclasses import dataclass  # Удобная структура для строк SRT
from pathlib import Path  # Работа с путями
from tempfile import TemporaryDirectory  # Временная папка для подготовленного WAV

import whisper  # Whisper для транскрибации


# Какие файлы считаем аудио
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

# Параметры по умолчанию
DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "ru"

# В какой формат приводим аудио перед распознаванием
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


@dataclass
class TranscriptLine:
    # Одна реплика для SRT
    start: float
    end: float
    text: str


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    # Запускаем внешнюю команду
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Если команда завершилась с ошибкой — показываем подробности
    if result.returncode != 0:
        raise RuntimeError(
            "Ошибка при выполнении команды:\n"
            f"{' '.join(cmd)}\n\n"
            f"{result.stderr}"
        )

    return result


def ensure_ffmpeg_tools() -> None:
    # Проверяем, что ffmpeg установлен
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg не найден в PATH")

    # Проверяем, что ffprobe установлен
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe не найден в PATH")


def get_audio_duration_seconds(input_file: Path) -> float:
    # Узнаём длительность файла через ffprobe
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(input_file),
    ]
    result = run_command(cmd)
    data = json.loads(result.stdout)

    duration = float(data["format"]["duration"])
    if duration <= 0:
        raise RuntimeError(f"Не удалось определить длительность файла: {input_file}")

    return duration


def format_srt_timestamp(seconds: float) -> str:
    # Переводим секунды в формат SRT: HH:MM:SS,mmm
    if seconds < 0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(lines: list[TranscriptLine], output_file: Path) -> None:
    # Сохраняем субтитры в SRT
    with open(output_file, "w", encoding="utf-8") as f:
        index = 1

        for line in lines:
            text = line.text.strip()
            if not text:
                continue

            f.write(f"{index}\n")
            f.write(
                f"{format_srt_timestamp(line.start)} --> "
                f"{format_srt_timestamp(line.end)}\n"
            )
            f.write(text + "\n\n")

            index += 1


def prepare_audio_for_whisper(input_file: Path, output_wav: Path) -> None:
    # Приводим исходное аудио к стабильному формату:
    # mono, 16 kHz, WAV PCM.
    #
    # Это уменьшает сюрпризы от разных кодеков и контейнеров.
    # Фильтры мягкие:
    # - highpass=80 убирает низкочастотный гул
    # - lowpass=7600 чуть чистит самый верх
    # - loudnorm слегка выравнивает громкость
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-af", "highpass=f=80,lowpass=f=7600,loudnorm",
        str(output_wav),
    ]
    run_command(cmd)


def transcribe_audio_file(
        model,
        prepared_audio_file: Path,
        language: str,
        beam_size: int,
) -> tuple[str, list[TranscriptLine]]:
    # Транскрибируем файл целиком.
    #
    # Это ключевое отличие:
    # модель видит весь контекст сразу,
    # поэтому часто лучше держит смысл, согласование и фразы.
    result = model.transcribe(
        str(prepared_audio_file),
        language=language,
        task="transcribe",
        fp16=False,
        temperature=0.0,
        beam_size=beam_size,
        best_of=1,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.5,
        verbose=False,
    )

    full_text = result.get("text", "").strip()
    srt_lines: list[TranscriptLine] = []

    for seg in result.get("segments", []):
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        srt_lines.append(
            TranscriptLine(
                start=start,
                end=end,
                text=seg_text,
            )
        )

    return full_text, srt_lines


def process_one_audio_file(
        audio_file: Path,
        model,
        language: str,
        beam_size: int,
) -> None:
    print("=" * 80)
    print(f"Обрабатываю файл: {audio_file.name}")
    print("=" * 80)

    total_duration = get_audio_duration_seconds(audio_file)
    print(f"Длительность: {total_duration:.2f} сек")

    # Создаём папку результатов рядом с исходным файлом
    output_dir = audio_file.parent / f"{audio_file.stem}_transcript"
    output_dir.mkdir(exist_ok=True)

    final_txt = output_dir / f"{audio_file.stem}.txt"
    final_srt = output_dir / f"{audio_file.stem}.srt"

    # Делаем один временный WAV для стабильной подачи в Whisper
    with TemporaryDirectory(prefix="whisper_prepare_") as tmp_dir:
        prepared_wav = Path(tmp_dir) / f"{audio_file.stem}_prepared.wav"

        print("Подготавливаю аудио...")
        prepare_audio_for_whisper(audio_file, prepared_wav)

        print("Транскрибирую целиком...")
        full_text, srt_lines = transcribe_audio_file(
            model=model,
            prepared_audio_file=prepared_wav,
            language=language,
            beam_size=beam_size,
        )

    # Сохраняем общий текст
    with open(final_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Сохраняем общий SRT
    write_srt(srt_lines, final_srt)

    print(f"TXT сохранён: {final_txt}")
    print(f"SRT сохранён: {final_srt}")
    print()


def parse_args() -> argparse.Namespace:
    # Разбираем аргументы командной строки
    parser = argparse.ArgumentParser(
        description="Транскрибация аудиофайлов через Whisper без разбиения на чанки"
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Путь к папке с аудиофайлами",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Модель Whisper: tiny, base, small, medium, large",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Язык речи, например ru, en, de",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Ширина beam search для Whisper",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)

    # Проверяем папку
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Ошибка: папка не найдена: {input_dir}")
        raise SystemExit(1)

    # Проверяем ffmpeg заранее
    ensure_ffmpeg_tools()

    # Ищем все подходящие аудиофайлы
    audio_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print("В указанной папке нет подходящих аудиофайлов.")
        raise SystemExit(0)

    print(f"Загружаю модель Whisper {args.model}...")
    model = whisper.load_model(args.model)
    print("Модель загружена.")
    print()

    success_count = 0
    failed_files: list[str] = []

    for audio_file in audio_files:
        try:
            process_one_audio_file(
                audio_file=audio_file,
                model=model,
                language=args.language,
                beam_size=args.beam_size,
            )
            success_count += 1
        except Exception as e:
            print(f"Ошибка при обработке файла {audio_file.name}: {e}")
            print()
            failed_files.append(audio_file.name)

    print("=" * 80)
    print("Обработка завершена.")
    print(f"Успешно обработано файлов: {success_count} из {len(audio_files)}")

    if failed_files:
        print("Не удалось обработать:")
        for name in failed_files:
            print(f" - {name}")


if __name__ == "__main__":
    main()