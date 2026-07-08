import argparse
import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import whisper


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "ru"

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

CleanMode = Literal["conservative", "balanced", "aggressive"]


DEFAULT_INITIAL_PROMPT = (
    "Это русскоязычное учебное занятие. Возможны термины: учитель, ученик, "
    "задача, условие, формула, решение, ответ, уравнение, неравенство, "
    "логарифм, производная, функция, график, степень, корень, дискриминант, "
    "физический смысл, ЕГЭ, ОГЭ, математика, физика, химия."
)


@dataclass
class TranscriptLine:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


IMPORTANT_STUDENT_SIGNALS = [
    "не понимаю",
    "не поняла",
    "не понял",
    "можно еще раз",
    "можно ещё раз",
    "повторите",
    "почему",
    "я зависла",
    "я завис",
    "не получается",
    "у меня не так",
    "у меня другой ответ",
    "я получила другой ответ",
    "я получил другой ответ",
    "не сходится",
    "я запуталась",
    "я запутался",
    "что это такое",
]


REPEATABLE_PHRASES = [
    "угу",
    "ага",
    "да",
    "нет",
    "спасибо",
    "повторяйте",
    "я не знаю",
    "я не знаю, что это такое",
    "слышишь",
    "есть",
]


SUBTITLE_ARTIFACT_PATTERNS = [
    r"Редактор\s+субтитров\s+[А-ЯЁA-Z]\.\s?[А-ЯЁа-яёA-Za-z-]+",
    r"Корректор\s+[А-ЯЁA-Z]\.\s?[А-ЯЁа-яёA-Za-z-]+",
    r"Редактор\s+субтитров\s+[А-ЯЁа-яёA-Za-z-]+",
    r"Корректор\s+[А-ЯЁа-яёA-Za-z-]+",
    r"Субтитры\s+подготовлены[^.!?\n]{0,120}",
    r"Субтитры\s+созданы[^.!?\n]{0,120}",
    r"Спасибо\s+за\s+просмотр[.!?]?",
    r"Продолжение\s+следует[.!?]?",
]


FILLER_SENTENCE_PATTERN = re.compile(
    r"^(?:"
    r"так|ну|угу|ага|да|нет|отлично|хорошо|ладно|секундочку|"
    r"сейчас|все|всё|понятно|готовы|согласны|окей|ок|"
    r"давайте|супер|нормально"
    r")[\s,!.?…-]*$",
    flags=re.IGNORECASE,
)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    logging.debug("Запуск команды: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Ошибка при выполнении команды:\n"
            f"{' '.join(cmd)}\n\n"
            f"{result.stderr}"
        )

    return result


def ensure_ffmpeg_tools() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установите ffmpeg и добавьте его в PATH."
        )

    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe не найден в PATH. Обычно он устанавливается вместе с ffmpeg."
        )


def get_audio_duration_seconds(input_file: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(input_file),
    ]

    result = run_command(cmd)
    data = json.loads(result.stdout)

    try:
        duration = float(data["format"]["duration"])
    except KeyError as exc:
        raise RuntimeError(f"ffprobe не вернул длительность файла: {input_file}") from exc

    if duration <= 0:
        raise RuntimeError(f"Не удалось определить длительность файла: {input_file}")

    return duration


def format_srt_timestamp(seconds: float) -> str:
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


def format_plain_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_srt(lines: list[TranscriptLine], output_file: Path) -> None:
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


def write_timestamped_txt(lines: list[TranscriptLine], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for line in lines:
            text = line.text.strip()
            if not text:
                continue

            start = format_plain_timestamp(line.start)
            end = format_plain_timestamp(line.end)

            f.write(f"[{start} — {end}] {text}\n")


def write_json(data: object, output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def prepare_audio_for_whisper(input_file: Path, output_wav: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-af",
        "highpass=f=80,lowpass=f=7600,loudnorm",
        str(output_wav),
    ]

    run_command(cmd)


def transcribe_audio_file(
        model,
        prepared_audio_file: Path,
        language: str,
        beam_size: int,
        initial_prompt: str | None,
        condition_on_previous_text: bool,
) -> tuple[str, list[TranscriptLine]]:
    result = model.transcribe(
        str(prepared_audio_file),
        language=language,
        task="transcribe",
        fp16=False,
        temperature=0.0,
        beam_size=beam_size,
        best_of=1,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
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

        srt_lines.append(
            TranscriptLine(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=seg_text,
                avg_logprob=to_optional_float(seg.get("avg_logprob")),
                no_speech_prob=to_optional_float(seg.get("no_speech_prob")),
                compression_ratio=to_optional_float(seg.get("compression_ratio")),
            )
        )

    return full_text, srt_lines


def to_optional_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[^\s\n])", r"\1 ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def remove_subtitle_artifacts(text: str) -> str:
    for pattern in SUBTITLE_ARTIFACT_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text


def remove_known_asr_noise_blocks(text: str) -> str:
    replacements = [
        (
            r"(?:Я не знаю, что это такое[.!?…,\s]*){2,}",
            "[технический шум автотранскрибации удалён] ",
        ),
        (
            r"(?:Спасибо[.!?…,\s]*){5,}",
            "[длинный повтор слова «спасибо» удалён] ",
        ),
        (
            r"(?:Повторяйте[.!?…,\s]*){3,}",
            "[технический повтор фразы «повторяйте» удалён] ",
        ),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def collapse_target_repetitions(text: str, min_repeats: int) -> str:
    for phrase in REPEATABLE_PHRASES:
        pattern = rf"(?:\b{re.escape(phrase)}\b[.!?…,\s]*){{{min_repeats},}}"

        text = re.sub(
            pattern,
            f"[повтор фразы «{phrase}» удалён] ",
            text,
            flags=re.IGNORECASE,
        )

    return text


def collapse_repeated_single_words(text: str, min_repeats: int) -> str:
    pattern = re.compile(
        rf"\b([А-Яа-яЁёA-Za-z0-9_-]{{2,}})\b"
        rf"(?:[,.!?…\s]+\1\b){{{min_repeats - 1},}}",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match) -> str:
        word = match.group(1)
        return f"[повтор слова «{word}» удалён]"

    return pattern.sub(replace, text)


def remove_aggressive_filler_sentences(text: str) -> str:
    sentences = split_sentences(text)
    kept: list[str] = []

    for sentence in sentences:
        stripped = sentence.strip()

        if not stripped:
            continue

        if FILLER_SENTENCE_PATTERN.match(stripped):
            continue

        kept.append(stripped)

    return " ".join(kept)


def split_sentences(text: str) -> list[str]:
    # Простое универсальное разбиение.
    # Для учебных транскриптов этого обычно достаточно перед LLM-этапом.
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def extract_important_student_signals(text: str) -> list[dict[str, object]]:
    signals: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for signal in IMPORTANT_STUDENT_SIGNALS:
        pattern = re.compile(re.escape(signal), flags=re.IGNORECASE)

        for match in pattern.finditer(text):
            start = max(0, match.start() - 120)
            end = min(len(text), match.end() + 120)
            snippet = text[start:end].strip()

            key = (signal.lower(), snippet.lower())
            if key in seen:
                continue

            seen.add(key)

            signals.append(
                {
                    "signal": signal,
                    "position": match.start(),
                    "snippet": snippet,
                }
            )

    return signals


def split_text_into_chunks(
        text: str,
        max_chars: int,
        overlap_chars: int,
) -> list[str]:
    if max_chars <= 1000:
        raise ValueError("chunk_size должен быть больше 1000 символов.")

    if overlap_chars < 0:
        raise ValueError("chunk_overlap не может быть отрицательным.")

    if len(text) <= max_chars:
        return [text] if text.strip() else []

    sentences = split_sentences(text)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1

        if current_parts and current_len + sentence_len > max_chars:
            chunk = " ".join(current_parts).strip()
            chunks.append(chunk)

            if overlap_chars > 0:
                overlap = chunk[-overlap_chars:].strip()
                current_parts = [overlap, sentence]
                current_len = len(overlap) + sentence_len + 1
            else:
                current_parts = [sentence]
                current_len = sentence_len
        else:
            current_parts.append(sentence)
            current_len += sentence_len

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    return chunks


def clean_transcript_text(raw_text: str, mode: CleanMode) -> dict[str, object]:
    normalized = normalize_text(raw_text)

    cleaned = remove_subtitle_artifacts(normalized)

    if mode == "conservative":
        cleaned = collapse_target_repetitions(cleaned, min_repeats=8)
        cleaned = collapse_repeated_single_words(cleaned, min_repeats=8)

    elif mode == "balanced":
        cleaned = remove_known_asr_noise_blocks(cleaned)
        cleaned = collapse_target_repetitions(cleaned, min_repeats=4)
        cleaned = collapse_repeated_single_words(cleaned, min_repeats=5)

    elif mode == "aggressive":
        cleaned = remove_known_asr_noise_blocks(cleaned)
        cleaned = collapse_target_repetitions(cleaned, min_repeats=3)
        cleaned = collapse_repeated_single_words(cleaned, min_repeats=4)
        cleaned = remove_aggressive_filler_sentences(cleaned)

    else:
        raise ValueError(f"Неизвестный режим очистки: {mode}")

    cleaned = normalize_text(cleaned)
    signals = extract_important_student_signals(cleaned)

    return {
        "normalized": normalized,
        "cleaned": cleaned,
        "signals": signals,
    }


def create_llm_prompt(
        cleaned_text: str,
        signals: list[dict[str, object]],
        chunk_count: int,
        include_text_limit: int,
) -> str:
    signal_lines = []

    for item in signals[:30]:
        signal = item["signal"]
        snippet = item["snippet"]
        signal_lines.append(f"- `{signal}`: {snippet}")

    if not signal_lines:
        signal_block = "Явные сигналы затруднений ученика автоматически не найдены."
    else:
        signal_block = "\n".join(signal_lines)

    if len(cleaned_text) <= include_text_limit:
        transcript_block = cleaned_text
    else:
        transcript_block = (
            "[Транскрипт слишком длинный для одного prompt-файла. "
            "Используйте файлы из папки 03_llm_chunks и обрабатывайте их по очереди.]"
        )

    return f"""Ты — редактор учебных транскриптов, методист и предметный аналитик.

Я дам тебе очищенный, но всё ещё сырой транскрипт занятия.

Твоя задача — подготовить его для дальнейшей генерации учебных материалов.

Работай по правилам:

1. Удали остаточный технический мусор:
   - субтитровые вставки;
   - случайные повторы;
   - сбои автотранскрибации;
   - бессмысленные повторяющиеся фразы.

2. Не удаляй педагогически значимые фрагменты:
   - вопросы ученика;
   - ошибки ученика;
   - затруднения;
   - просьбы повторить;
   - неуверенные ответы;
   - методические акценты учителя.

3. Раздели занятие на смысловые блоки:
   - организационное начало;
   - постановка темы;
   - задача;
   - объяснение;
   - самостоятельная работа;
   - разбор ошибки;
   - итог;
   - домашнее задание.

4. Восстанови формулы в LaTeX, если это возможно.
   Если формула восстановлена неуверенно, обязательно добавь пометку:
   [требуется проверка по исходному условию].

5. Не выдумывай условия задач, данные и ответы.
   Если фрагмент неясен, пометь:
   [неясно].

6. Выдели:
   - темы занятия;
   - ключевые формулы;
   - разобранные задачи;
   - основные приёмы;
   - ошибки и затруднения ученика;
   - методические выводы;
   - материал, который стоит включить в пособие.

Формат результата:

# Очищенный учебный протокол занятия

## 1. Краткая сводка

## 2. Темы занятия

## 3. Смысловые блоки

## 4. Разобранные задачи

Для каждой задачи:
- название/тема;
- условие, если восстановимо;
- данные;
- формула;
- ход решения;
- ответ;
- методический смысл;
- потенциальные ошибки.

## 5. Ошибки и затруднения ученика

## 6. Важные методические акценты

## 7. Что включить в персональное пособие

## 8. Неясные места

---

# Автоматически найденные сигналы ученика

{signal_block}

---

# Количество чанков

{chunk_count}

---

# Очищенный транскрипт

{transcript_block}
"""


def write_cleaning_report(
        output_file: Path,
        audio_file: Path,
        raw_text: str,
        cleaned_text: str,
        mode: CleanMode,
        chunks: list[str],
        signals: list[dict[str, object]],
) -> None:
    report = f"""# Отчёт об очистке транскрипта

## Файл

`{audio_file.name}`

## Режим очистки

`{mode}`

## Размеры текста

- Сырой текст: {len(raw_text)} символов
- Очищенный текст: {len(cleaned_text)} символов
- Чанков для LLM: {len(chunks)}

## Найденные важные сигналы ученика

"""

    if signals:
        for item in signals[:50]:
            report += f"- **{item['signal']}**: {item['snippet']}\n"
    else:
        report += "Явные сигналы затруднений ученика автоматически не найдены.\n"

    report += """

## Что делать дальше

1. Открыть `02_clean_*.txt` и быстро проверить качество очистки.
2. Если текст слишком очищен, повторить обработку с режимом `conservative`.
3. Если мусора осталось много, повторить с режимом `aggressive`.
4. Для создания пособия использовать `04_llm_prompt.md` или чанки из папки `03_llm_chunks`.
"""

    output_file.write_text(report, encoding="utf-8")


def process_prepared_transcript(
        raw_text: str,
        audio_file: Path,
        output_dir: Path,
        clean_mode: CleanMode,
        chunk_size: int,
        chunk_overlap: int,
        prompt_include_text_limit: int,
) -> dict[str, object]:
    cleaned_result = clean_transcript_text(raw_text, mode=clean_mode)

    normalized_text = str(cleaned_result["normalized"])
    cleaned_text = str(cleaned_result["cleaned"])
    signals = list(cleaned_result["signals"])

    chunks = split_text_into_chunks(
        cleaned_text,
        max_chars=chunk_size,
        overlap_chars=chunk_overlap,
    )

    normalized_file = output_dir / "01_normalized.txt"
    clean_file = output_dir / f"02_clean_{clean_mode}.txt"
    chunks_dir = output_dir / "03_llm_chunks"
    prompt_file = output_dir / "04_llm_prompt.md"
    signals_file = output_dir / "important_student_signals.json"
    report_file = output_dir / "cleaning_report.md"

    normalized_file.write_text(normalized_text, encoding="utf-8")
    clean_file.write_text(cleaned_text, encoding="utf-8")

    chunks_dir.mkdir(exist_ok=True)

    for index, chunk in enumerate(chunks, start=1):
        chunk_file = chunks_dir / f"chunk_{index:03}.txt"
        chunk_file.write_text(chunk, encoding="utf-8")

    prompt_text = create_llm_prompt(
        cleaned_text=cleaned_text,
        signals=signals,
        chunk_count=len(chunks),
        include_text_limit=prompt_include_text_limit,
    )
    prompt_file.write_text(prompt_text, encoding="utf-8")

    write_json(signals, signals_file)

    write_cleaning_report(
        output_file=report_file,
        audio_file=audio_file,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        mode=clean_mode,
        chunks=chunks,
        signals=signals,
    )

    return {
        "normalized_file": str(normalized_file),
        "clean_file": str(clean_file),
        "chunks_dir": str(chunks_dir),
        "prompt_file": str(prompt_file),
        "signals_file": str(signals_file),
        "report_file": str(report_file),
        "chunk_count": len(chunks),
        "student_signal_count": len(signals),
    }


def process_one_audio_file(
        audio_file: Path,
        model,
        language: str,
        beam_size: int,
        clean_mode: CleanMode,
        chunk_size: int,
        chunk_overlap: int,
        prompt_include_text_limit: int,
        initial_prompt: str | None,
        condition_on_previous_text: bool,
        keep_prepared_wav: bool,
) -> None:
    logging.info("=" * 80)
    logging.info("Обрабатываю файл: %s", audio_file.name)
    logging.info("=" * 80)

    total_duration = get_audio_duration_seconds(audio_file)
    logging.info("Длительность: %.2f сек", total_duration)

    output_dir = audio_file.parent / f"{audio_file.stem}_lesson_transcript"
    output_dir.mkdir(exist_ok=True)

    raw_txt = output_dir / "00_raw_whisper.txt"
    raw_timestamped_txt = output_dir / "00_raw_timestamped.txt"
    raw_srt = output_dir / "00_raw_segments.srt"
    raw_segments_json = output_dir / "00_raw_segments.json"
    manifest_file = output_dir / "manifest.json"

    prepared_wav_to_keep = output_dir / "prepared_for_whisper.wav"

    with TemporaryDirectory(prefix="whisper_prepare_") as tmp_dir:
        prepared_wav = Path(tmp_dir) / f"{audio_file.stem}_prepared.wav"

        logging.info("Подготавливаю аудио...")
        prepare_audio_for_whisper(audio_file, prepared_wav)

        if keep_prepared_wav:
            shutil.copy2(prepared_wav, prepared_wav_to_keep)

        logging.info("Транскрибирую целиком...")
        full_text, srt_lines = transcribe_audio_file(
            model=model,
            prepared_audio_file=prepared_wav,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )

    raw_txt.write_text(full_text, encoding="utf-8")
    write_timestamped_txt(srt_lines, raw_timestamped_txt)
    write_srt(srt_lines, raw_srt)
    write_json([asdict(line) for line in srt_lines], raw_segments_json)

    logging.info("Провожу предобработку транскрипта...")
    preprocess_result = process_prepared_transcript(
        raw_text=full_text,
        audio_file=audio_file,
        output_dir=output_dir,
        clean_mode=clean_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        prompt_include_text_limit=prompt_include_text_limit,
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_audio": str(audio_file),
        "output_dir": str(output_dir),
        "duration_seconds": total_duration,
        "whisper": {
            "language": language,
            "beam_size": beam_size,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt_used": bool(initial_prompt),
        },
        "audio_prepare": {
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "target_channels": TARGET_CHANNELS,
            "keep_prepared_wav": keep_prepared_wav,
        },
        "preprocessing": {
            "clean_mode": clean_mode,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "prompt_include_text_limit": prompt_include_text_limit,
        },
        "outputs": {
            "raw_txt": str(raw_txt),
            "raw_timestamped_txt": str(raw_timestamped_txt),
            "raw_srt": str(raw_srt),
            "raw_segments_json": str(raw_segments_json),
            **preprocess_result,
        },
    }

    write_json(manifest, manifest_file)

    logging.info("Готово.")
    logging.info("Папка результата: %s", output_dir)
    logging.info("Сырой TXT: %s", raw_txt.name)
    logging.info("Очищенный TXT: 02_clean_%s.txt", clean_mode)
    logging.info("Prompt для LLM: 04_llm_prompt.md")
    logging.info("Чанков для LLM: %s", preprocess_result["chunk_count"])
    logging.info("Сигналов ученика найдено: %s", preprocess_result["student_signal_count"])
    logging.info("")


def collect_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.iterdir()

    return sorted(
        file
        for file in candidates
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Транскрибация аудиофайлов через Whisper с последующей "
            "универсальной предобработкой учебных транскриптов."
        )
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Путь к папке с аудиофайлами.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Модель Whisper: tiny, base, small, medium, large.",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Язык речи, например ru, en, de.",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Ширина beam search для Whisper.",
    )

    parser.add_argument(
        "--clean-mode",
        type=str,
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help=(
            "Режим очистки транскрипта: "
            "conservative — бережный, "
            "balanced — рабочий, "
            "aggressive — сильная очистка."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=6000,
        help="Максимальный размер одного чанка для LLM в символах.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=500,
        help="Перекрытие между чанками в символах.",
    )

    parser.add_argument(
        "--prompt-include-text-limit",
        type=int,
        default=30000,
        help=(
            "До какого размера очищенный транскрипт целиком вставляется "
            "в 04_llm_prompt.md. Если текст длиннее, используйте чанки."
        ),
    )

    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=DEFAULT_INITIAL_PROMPT,
        help=(
            "Начальная подсказка для Whisper. "
            "Можно передать пустую строку, чтобы отключить."
        ),
    )

    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help=(
            "Включить condition_on_previous_text=True. "
            "Иногда улучшает связность, но может усиливать повторы и галлюцинации."
        ),
    )

    parser.add_argument(
        "--keep-prepared-wav",
        action="store_true",
        help="Сохранять подготовленный WAV рядом с результатами.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Искать аудиофайлы во вложенных папках.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный лог.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    input_dir = Path(args.input_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        logging.error("Папка не найдена: %s", input_dir)
        raise SystemExit(1)

    ensure_ffmpeg_tools()

    audio_files = collect_audio_files(
        input_dir=input_dir,
        recursive=args.recursive,
    )

    if not audio_files:
        logging.info("В указанной папке нет подходящих аудиофайлов.")
        raise SystemExit(0)

    initial_prompt = args.initial_prompt.strip()
    if not initial_prompt:
        initial_prompt = None

    logging.info("Найдено аудиофайлов: %s", len(audio_files))
    logging.info("Загружаю модель Whisper: %s", args.model)

    model = whisper.load_model(args.model)

    logging.info("Модель загружена.")
    logging.info("")

    success_count = 0
    failed_files: list[str] = []

    for audio_file in audio_files:
        try:
            process_one_audio_file(
                audio_file=audio_file,
                model=model,
                language=args.language,
                beam_size=args.beam_size,
                clean_mode=args.clean_mode,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                prompt_include_text_limit=args.prompt_include_text_limit,
                initial_prompt=initial_prompt,
                condition_on_previous_text=args.condition_on_previous_text,
                keep_prepared_wav=args.keep_prepared_wav,
            )
            success_count += 1

        except KeyboardInterrupt:
            logging.warning("Обработка прервана пользователем.")
            raise SystemExit(130)

        except Exception as exc:
            logging.exception("Ошибка при обработке файла %s: %s", audio_file.name, exc)
            failed_files.append(audio_file.name)

    logging.info("=" * 80)
    logging.info("Обработка завершена.")
    logging.info("Успешно обработано файлов: %s из %s", success_count, len(audio_files))

    if failed_files:
        logging.info("Не удалось обработать:")
        for name in failed_files:
            logging.info(" - %s", name)


if __name__ == "__main__":
    main()