from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal
from urllib import error as url_error
from urllib import request as url_request

from faster_whisper import WhisperModel


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "ru"

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

CleanMode = Literal["conservative", "balanced", "aggressive"]
AudioFilterMode = Literal["none", "basic", "loudnorm"]


DEFAULT_INITIAL_PROMPT = (
    "Это русскоязычное учебное занятие. Возможны термины: учитель, ученик, "
    "задача, условие, формула, решение, ответ, уравнение, неравенство, "
    "логарифм, производная, функция, график, степень, корень, дискриминант, "
    "физический смысл, ЕГЭ, ОГЭ, математика, физика, химия."
)


OLLAMA_SYSTEM_PROMPT = """Ты — опытный методист, редактор учебных транскриптов и предметный аналитик.

Твоя задача — превращать очищенные фрагменты занятий в аккуратный учебный протокол.

Правила:
1. Не выдумывай условия задач, числа, ответы и формулы.
2. Если формула или условие восстановлены неуверенно, пиши: [требуется проверка].
3. Если фрагмент неясен, пиши: [неясно].
4. Сохраняй вопросы, ошибки, затруднения и неуверенность ученика.
5. Не делай вывод «ученик понял», если это явно не подтверждено.
6. Математические и физические формулы оформляй в LaTeX.
7. Отделяй реальные ошибки ученика от технического мусора транскрипции.
8. Пиши по-русски, структурированно, без воды.
"""


@dataclass
class TranscriptLine:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


@dataclass
class FasterWhisperConfig:
    model_name: str
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int
    beam_size: int
    language: str
    initial_prompt: str | None
    condition_on_previous_text: bool
    vad_filter: bool
    vad_min_silence_duration_ms: int
    vad_speech_pad_ms: int
    word_timestamps: bool


@dataclass
class OllamaConfig:
    enabled: bool
    base_url: str
    model: str
    timeout: int
    temperature: float
    num_ctx: int
    num_predict: int
    keep_alive: str
    final_synthesis: bool
    final_max_chars: int


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

    logging.getLogger("faster_whisper").setLevel(logging.DEBUG if verbose else logging.WARNING)


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


def has_ffmpeg_tools() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def ensure_ffmpeg_tools() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Он нужен только если не указан --skip-audio-prepare."
        )

    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe не найден в PATH. Обычно он устанавливается вместе с ffmpeg."
        )


def get_audio_duration_seconds(input_file: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        logging.warning("ffprobe не найден: длительность до транскрибации не будет определена.")
        return None

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
    except KeyError:
        logging.warning("ffprobe не вернул длительность файла: %s", input_file)
        return None

    if duration <= 0:
        logging.warning("ffprobe вернул некорректную длительность файла: %s", input_file)
        return None

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


def build_audio_filter(mode: AudioFilterMode) -> str | None:
    if mode == "none":
        return None

    if mode == "basic":
        return "highpass=f=80,lowpass=f=7600"

    if mode == "loudnorm":
        return "highpass=f=80,lowpass=f=7600,loudnorm"

    raise ValueError(f"Неизвестный режим аудиофильтра: {mode}")


def prepare_audio_for_whisper(
        input_file: Path,
        output_wav: Path,
        audio_filter: AudioFilterMode,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-i",
        str(input_file),
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
    ]

    filter_chain = build_audio_filter(audio_filter)
    if filter_chain:
        cmd.extend(["-af", filter_chain])

    cmd.append(str(output_wav))

    run_command(cmd)


def to_optional_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_default_cpu_threads() -> int:
    cpu_count = os.cpu_count() or 4

    if cpu_count <= 2:
        return cpu_count

    return max(1, cpu_count - 1)


def apply_cpu_thread_environment(cpu_threads: int) -> None:
    if cpu_threads <= 0:
        return

    # CTranslate2/faster-whisper and other CPU backends may respect OMP_NUM_THREADS.
    # Не перезаписываем переменную, если пользователь уже настроил её вручную.
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))


def load_faster_whisper_model(config: FasterWhisperConfig) -> WhisperModel:
    apply_cpu_thread_environment(config.cpu_threads)

    logging.info(
        "Загружаю faster-whisper: model=%s, device=%s, compute_type=%s, cpu_threads=%s, num_workers=%s",
        config.model_name,
        config.device,
        config.compute_type,
        config.cpu_threads,
        config.num_workers,
    )

    model = WhisperModel(
        config.model_name,
        device=config.device,
        compute_type=config.compute_type,
        cpu_threads=config.cpu_threads,
        num_workers=config.num_workers,
    )

    return model


def transcribe_audio_file(
        model: WhisperModel,
        audio_file: Path,
        config: FasterWhisperConfig,
) -> tuple[str, list[TranscriptLine], dict[str, object]]:
    vad_parameters = {
        "min_silence_duration_ms": config.vad_min_silence_duration_ms,
        "speech_pad_ms": config.vad_speech_pad_ms,
    }

    segments_iter, info = model.transcribe(
        str(audio_file),
        language=config.language,
        task="transcribe",
        beam_size=config.beam_size,
        temperature=0.0,
        initial_prompt=config.initial_prompt,
        condition_on_previous_text=config.condition_on_previous_text,
        vad_filter=config.vad_filter,
        vad_parameters=vad_parameters if config.vad_filter else None,
        word_timestamps=config.word_timestamps,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.5,
    )

    lines: list[TranscriptLine] = []
    full_text_parts: list[str] = []

    # В faster-whisper segments — генератор; реальная транскрибация начинается здесь.
    for seg in segments_iter:
        seg_text = str(getattr(seg, "text", "")).strip()
        if not seg_text:
            continue

        full_text_parts.append(seg_text)

        lines.append(
            TranscriptLine(
                start=float(getattr(seg, "start", 0.0)),
                end=float(getattr(seg, "end", 0.0)),
                text=seg_text,
                avg_logprob=to_optional_float(getattr(seg, "avg_logprob", None)),
                no_speech_prob=to_optional_float(getattr(seg, "no_speech_prob", None)),
                compression_ratio=to_optional_float(getattr(seg, "compression_ratio", None)),
            )
        )

    full_text = " ".join(full_text_parts).strip()

    transcription_info = {
        "language": getattr(info, "language", None),
        "language_probability": to_optional_float(getattr(info, "language_probability", None)),
        "duration": to_optional_float(getattr(info, "duration", None)),
        "duration_after_vad": to_optional_float(getattr(info, "duration_after_vad", None)),
        "all_language_probs": getattr(info, "all_language_probs", None),
    }

    return full_text, lines, transcription_info


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


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [part.strip() for part in parts if part.strip()]


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

    for old_chunk in chunks_dir.glob("chunk_*.txt"):
        old_chunk.unlink()

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


def normalize_ollama_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def ollama_api_url(config: OllamaConfig, path: str) -> str:
    base_url = normalize_ollama_base_url(config.base_url)
    path = path if path.startswith("/") else f"/{path}"
    return f"{base_url}{path}"


def ollama_request_json(
        url: str,
        payload: dict[str, object] | None,
        timeout: int,
        method: str = "POST",
) -> dict[str, object]:
    data = None

    headers = {
        "Content-Type": "application/json",
    }

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = url_request.Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with url_request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")

    except url_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama вернул HTTP {exc.code}:\n{error_body}"
        ) from exc

    except url_error.URLError as exc:
        raise RuntimeError(
            "Не удалось подключиться к Ollama.\n"
            f"URL: {url}\n"
            "Проверьте, что Ollama запущена и доступна."
        ) from exc

    try:
        return json.loads(raw_body)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ollama вернула невалидный JSON:\n"
            f"{raw_body[:1000]}"
        ) from exc


def check_ollama_available(config: OllamaConfig) -> None:
    url = ollama_api_url(config, "/api/tags")

    response = ollama_request_json(
        url=url,
        payload=None,
        timeout=config.timeout,
        method="GET",
    )

    models = response.get("models", [])
    model_names: list[str] = []

    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and "name" in item:
                model_names.append(str(item["name"]))

    if config.model not in model_names:
        available = "\n".join(f" - {name}" for name in model_names) or "модели не найдены"

        raise RuntimeError(
            f"Модель Ollama не найдена локально: {config.model}\n\n"
            "Доступные модели:\n"
            f"{available}\n\n"
            "Скачайте модель командой:\n"
            f"ollama pull {config.model}"
        )


def ollama_chat(
        config: OllamaConfig,
        system_prompt: str,
        user_prompt: str,
) -> tuple[str, dict[str, object]]:
    url = ollama_api_url(config, "/api/chat")

    payload: dict[str, object] = {
        "model": config.model,
        "stream": False,
        "keep_alive": config.keep_alive,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "options": {
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "num_predict": config.num_predict,
        },
    }

    response = ollama_request_json(
        url=url,
        payload=payload,
        timeout=config.timeout,
        method="POST",
    )

    message = response.get("message", {})

    if not isinstance(message, dict):
        raise RuntimeError(f"Неожиданный формат ответа Ollama: {response}")

    content = str(message.get("content", "")).strip()

    if not content:
        raise RuntimeError(
            "Ollama вернула пустой ответ.\n"
            f"Полный ответ:\n{json.dumps(response, ensure_ascii=False, indent=2)}"
        )

    return content, response


def build_ollama_chunk_prompt(
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
) -> str:
    return f"""Перед тобой очищенный фрагмент транскрипта учебного занятия.

Это чанк {chunk_index} из {total_chunks}.

Твоя задача — превратить этот фрагмент в методически полезный протокол.

Выведи результат строго в формате Markdown:

# Протокол фрагмента {chunk_index}

## 1. Краткое содержание фрагмента

## 2. Темы и подтемы

## 3. Разобранные задачи

Для каждой задачи укажи:
- тема;
- условие, если восстановимо;
- данные;
- формула;
- ход решения;
- ответ, если он есть;
- что важно запомнить.

## 4. Формулы и математические объекты

## 5. Ошибки и затруднения ученика

Разделяй:
- явные ошибки;
- вероятные ошибки;
- затруднения;
- неясные места из-за качества транскрипта.

## 6. Методические акценты учителя

## 7. Что можно включить в персональное пособие

## 8. Неясные места

Правила:
- не выдумывай недостающие условия;
- сомнительные формулы помечай как [требуется проверка];
- если фрагмент является техническим шумом, так и напиши;
- сохраняй важные признаки непонимания ученика.

---

# Фрагмент транскрипта

{chunk_text}
"""


def build_ollama_final_synthesis_prompt(chunk_protocols: list[str]) -> str:
    joined = "\n\n---\n\n".join(chunk_protocols)

    return f"""Перед тобой протоколы отдельных фрагментов одного учебного занятия.

Твоя задача — собрать из них единый учебный протокол занятия.

Не выдумывай данные, задачи, формулы и ответы. Если в протоколах чанков есть противоречия, явно укажи это в разделе "Неясные места".

Выведи результат строго в Markdown:

# Итоговый учебный протокол занятия

## 1. Краткая сводка занятия

## 2. Основные темы занятия

## 3. Хронология занятия

## 4. Разобранные задачи

Для каждой задачи:
- название или тема;
- восстановленное условие, если возможно;
- данные;
- формула;
- решение;
- ответ;
- методический смысл;
- потенциальные ошибки.

## 5. Ключевые формулы

## 6. Основные приёмы

## 7. Ошибки и затруднения ученика

Раздели:
- явные ошибки;
- вероятные ошибки;
- затруднения;
- места, где ученик просил помощи;
- места, где качество транскрипта мешает выводу.

## 8. Что стоит включить в персональное пособие

## 9. Рекомендуемый тренировочный блок

## 10. Неясные места, требующие проверки

---

# Протоколы фрагментов

{joined}
"""


def run_ollama_processing_for_lesson(
        output_dir: Path,
        preprocess_result: dict[str, object],
        config: OllamaConfig,
) -> dict[str, object]:
    chunks_dir = Path(str(preprocess_result["chunks_dir"]))
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))

    if not chunk_files:
        raise RuntimeError(f"Не найдены чанки для Ollama: {chunks_dir}")

    ollama_chunks_dir = output_dir / "05_ollama_chunk_protocols"
    ollama_chunks_dir.mkdir(exist_ok=True)

    for old_file in ollama_chunks_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    report: dict[str, object] = {
        "enabled": config.enabled,
        "base_url": config.base_url,
        "model": config.model,
        "chunk_count": len(chunk_files),
        "chunk_outputs": [],
        "final_synthesis": config.final_synthesis,
    }

    chunk_protocols: list[str] = []

    logging.info("Проверяю доступность Ollama...")
    check_ollama_available(config)

    logging.info(
        "Запускаю Ollama-обработку чанков: модель %s, чанков %s",
        config.model,
        len(chunk_files),
    )

    chunk_outputs = []

    for index, chunk_file in enumerate(chunk_files, start=1):
        chunk_text = chunk_file.read_text(encoding="utf-8")

        user_prompt = build_ollama_chunk_prompt(
            chunk_text=chunk_text,
            chunk_index=index,
            total_chunks=len(chunk_files),
        )

        logging.info("Ollama: обрабатываю чанк %s/%s", index, len(chunk_files))

        started = perf_counter()
        content, raw_response = ollama_chat(
            config=config,
            system_prompt=OLLAMA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        elapsed = perf_counter() - started
        logging.info("Ollama: чанк %s/%s готов за %.2f сек", index, len(chunk_files), elapsed)

        output_file = ollama_chunks_dir / f"chunk_{index:03}_protocol.md"
        raw_response_file = ollama_chunks_dir / f"chunk_{index:03}_ollama_response.json"

        output_file.write_text(content, encoding="utf-8")
        write_json(raw_response, raw_response_file)

        chunk_protocols.append(content)

        chunk_outputs.append(
            {
                "chunk_file": str(chunk_file),
                "protocol_file": str(output_file),
                "raw_response_file": str(raw_response_file),
                "elapsed_seconds": round(elapsed, 3),
            }
        )

    report["chunk_outputs"] = chunk_outputs

    final_protocol_file = output_dir / "06_ollama_final_protocol.md"
    final_prompt_file = output_dir / "06_ollama_final_prompt.md"

    if config.final_synthesis:
        final_prompt = build_ollama_final_synthesis_prompt(chunk_protocols)

        if len(final_prompt) > config.final_max_chars:
            warning_text = (
                "# Итоговый протокол не был автоматически собран\n\n"
                f"Финальный prompt получился слишком большим: {len(final_prompt)} символов.\n\n"
                f"Ограничение: {config.final_max_chars} символов.\n\n"
                "Используйте протоколы чанков из папки `05_ollama_chunk_protocols` "
                "или увеличьте `--ollama-final-max-chars` и `--ollama-num-ctx`."
            )

            final_protocol_file.write_text(warning_text, encoding="utf-8")
            final_prompt_file.write_text(final_prompt, encoding="utf-8")

            report["final_protocol_file"] = str(final_protocol_file)
            report["final_prompt_file"] = str(final_prompt_file)
            report["final_status"] = "skipped_too_large"

        else:
            logging.info("Ollama: собираю итоговый протокол занятия...")

            started = perf_counter()
            final_content, final_raw_response = ollama_chat(
                config=config,
                system_prompt=OLLAMA_SYSTEM_PROMPT,
                user_prompt=final_prompt,
            )
            elapsed = perf_counter() - started
            logging.info("Ollama: итоговый протокол собран за %.2f сек", elapsed)

            final_protocol_file.write_text(final_content, encoding="utf-8")
            final_prompt_file.write_text(final_prompt, encoding="utf-8")

            final_response_file = output_dir / "06_ollama_final_response.json"
            write_json(final_raw_response, final_response_file)

            report["final_protocol_file"] = str(final_protocol_file)
            report["final_prompt_file"] = str(final_prompt_file)
            report["final_response_file"] = str(final_response_file)
            report["final_elapsed_seconds"] = round(elapsed, 3)
            report["final_status"] = "created"

    else:
        report["final_status"] = "disabled"

    report_file = output_dir / "ollama_report.json"
    write_json(report, report_file)

    report["report_file"] = str(report_file)

    logging.info("Ollama-этап завершён.")

    if final_protocol_file.exists():
        logging.info("Итоговый протокол: %s", final_protocol_file)

    return report


def process_one_audio_file(
        audio_file: Path,
        model: WhisperModel,
        whisper_config: FasterWhisperConfig,
        clean_mode: CleanMode,
        chunk_size: int,
        chunk_overlap: int,
        prompt_include_text_limit: int,
        keep_prepared_wav: bool,
        skip_audio_prepare: bool,
        audio_filter: AudioFilterMode,
        ollama_config: OllamaConfig | None,
) -> None:
    logging.info("=" * 80)
    logging.info("Обрабатываю файл: %s", audio_file.name)
    logging.info("=" * 80)

    total_started = perf_counter()
    total_duration = get_audio_duration_seconds(audio_file)
    if total_duration is not None:
        logging.info("Длительность: %.2f сек", total_duration)

    output_dir = audio_file.parent / f"{audio_file.stem}_lesson_transcript"
    output_dir.mkdir(exist_ok=True)

    raw_txt = output_dir / "00_raw_whisper.txt"
    raw_timestamped_txt = output_dir / "00_raw_timestamped.txt"
    raw_srt = output_dir / "00_raw_segments.srt"
    raw_segments_json = output_dir / "00_raw_segments.json"
    manifest_file = output_dir / "manifest.json"

    prepared_wav_to_keep = output_dir / "prepared_for_whisper.wav"

    transcription_info: dict[str, object] = {}
    prepare_time = 0.0
    whisper_time = 0.0

    if skip_audio_prepare:
        logging.info("Пропускаю ffmpeg-подготовку: faster-whisper будет читать исходный файл напрямую.")
        audio_for_transcription = audio_file

        logging.info("Транскрибирую через faster-whisper...")
        started = perf_counter()
        full_text, transcript_lines, transcription_info = transcribe_audio_file(
            model=model,
            audio_file=audio_for_transcription,
            config=whisper_config,
        )
        whisper_time = perf_counter() - started

    else:
        with TemporaryDirectory(prefix="whisper_prepare_") as tmp_dir:
            prepared_wav = Path(tmp_dir) / f"{audio_file.stem}_prepared.wav"

            logging.info("Подготавливаю аудио через ffmpeg: filter=%s", audio_filter)
            started = perf_counter()
            prepare_audio_for_whisper(audio_file, prepared_wav, audio_filter=audio_filter)
            prepare_time = perf_counter() - started
            logging.info("Подготовка аудио заняла: %.2f сек", prepare_time)

            if keep_prepared_wav:
                shutil.copy2(prepared_wav, prepared_wav_to_keep)

            logging.info("Транскрибирую через faster-whisper...")
            started = perf_counter()
            full_text, transcript_lines, transcription_info = transcribe_audio_file(
                model=model,
                audio_file=prepared_wav,
                config=whisper_config,
            )
            whisper_time = perf_counter() - started

    effective_duration = total_duration or to_optional_float(transcription_info.get("duration"))
    realtime_factor = None
    if effective_duration is not None and whisper_time > 0:
        realtime_factor = effective_duration / whisper_time

    logging.info(
        "faster-whisper завершил транскрибацию за %.2f сек%s",
        whisper_time,
        f"; скорость: {realtime_factor:.2f}x realtime" if realtime_factor else "",
    )

    raw_txt.write_text(full_text, encoding="utf-8")
    write_timestamped_txt(transcript_lines, raw_timestamped_txt)
    write_srt(transcript_lines, raw_srt)
    write_json([asdict(line) for line in transcript_lines], raw_segments_json)

    logging.info("Провожу предобработку транскрипта...")
    started = perf_counter()
    preprocess_result = process_prepared_transcript(
        raw_text=full_text,
        audio_file=audio_file,
        output_dir=output_dir,
        clean_mode=clean_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        prompt_include_text_limit=prompt_include_text_limit,
    )
    preprocessing_time = perf_counter() - started
    logging.info("Предобработка заняла: %.2f сек", preprocessing_time)

    ollama_result = None

    if ollama_config is not None and ollama_config.enabled:
        logging.info("Запускаю локальную LLM-обработку через Ollama...")

        started = perf_counter()
        ollama_result = run_ollama_processing_for_lesson(
            output_dir=output_dir,
            preprocess_result=preprocess_result,
            config=ollama_config,
        )
        ollama_time = perf_counter() - started
    else:
        ollama_time = 0.0

    total_time = perf_counter() - total_started

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_audio": str(audio_file),
        "output_dir": str(output_dir),
        "duration_seconds": effective_duration,
        "timings_seconds": {
            "audio_prepare": round(prepare_time, 3),
            "faster_whisper": round(whisper_time, 3),
            "preprocessing": round(preprocessing_time, 3),
            "ollama": round(ollama_time, 3),
            "total": round(total_time, 3),
            "realtime_factor": round(realtime_factor, 3) if realtime_factor else None,
        },
        "faster_whisper": {
            "model": whisper_config.model_name,
            "device": whisper_config.device,
            "compute_type": whisper_config.compute_type,
            "cpu_threads": whisper_config.cpu_threads,
            "num_workers": whisper_config.num_workers,
            "language": whisper_config.language,
            "beam_size": whisper_config.beam_size,
            "condition_on_previous_text": whisper_config.condition_on_previous_text,
            "initial_prompt_used": bool(whisper_config.initial_prompt),
            "vad_filter": whisper_config.vad_filter,
            "vad_min_silence_duration_ms": whisper_config.vad_min_silence_duration_ms,
            "vad_speech_pad_ms": whisper_config.vad_speech_pad_ms,
            "word_timestamps": whisper_config.word_timestamps,
            "transcription_info": transcription_info,
        },
        "audio_prepare": {
            "skip_audio_prepare": skip_audio_prepare,
            "audio_filter": audio_filter,
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
        "ollama": ollama_result,
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
    logging.info("Итоговое время файла: %.2f сек", total_time)

    if ollama_result:
        logging.info("Ollama-протокол: 06_ollama_final_protocol.md")

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
            "Транскрибация аудиофайлов через faster-whisper с последующей "
            "универсальной предобработкой учебных транскриптов и "
            "опциональной обработкой через Ollama."
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
        help=(
            "Модель faster-whisper: tiny, base, small, medium, large-v3, large-v3-turbo, turbo "
            "или путь к локальной CTranslate2-модели. Для CPU обычно начинайте со small."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Устройство для faster-whisper. Для ноутбука без NVIDIA используйте cpu.",
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help=(
            "Тип вычислений CTranslate2. Для CPU рекомендуется int8. "
            "Для CUDA обычно float16 или int8_float16."
        ),
    )

    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=resolve_default_cpu_threads(),
        help=(
            "Количество CPU-потоков для faster-whisper. "
            "По умолчанию: число логических ядер минус 1. Поставьте 0 для выбора CTranslate2."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Количество worker'ов faster-whisper. Для последовательной обработки файлов обычно 1.",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Язык речи, например ru, en, de. Явное указание языка экономит время на автоопределении.",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Ширина beam search. 1 — быстрее, 2–5 — потенциально точнее, но медленнее.",
    )

    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Включить VAD-фильтр faster-whisper для пропуска фрагментов без речи. По умолчанию включён.",
    )

    parser.add_argument(
        "--vad-min-silence-duration-ms",
        type=int,
        default=1000,
        help="Минимальная длительность тишины для VAD, мс. Больше — бережнее, меньше — быстрее.",
    )

    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=300,
        help="Отступ вокруг найденной речи для VAD, мс.",
    )

    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Сохранять пословные таймкоды. Медленнее и для текущего пайплайна обычно не нужно.",
    )

    parser.add_argument(
        "--skip-audio-prepare",
        action="store_true",
        help=(
            "Не создавать WAV через ffmpeg, а отдавать исходный файл напрямую в faster-whisper. "
            "Обычно быстрее. Фильтрация highpass/lowpass/loudnorm при этом не применяется."
        ),
    )

    parser.add_argument(
        "--audio-filter",
        type=str,
        choices=["none", "basic", "loudnorm"],
        default="basic",
        help=(
            "Фильтр ffmpeg при подготовке WAV: none — без фильтров, "
            "basic — highpass/lowpass, loudnorm — медленнее, но нормализует громкость."
        ),
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
        help="Сохранять подготовленный WAV рядом с результатами. Работает только без --skip-audio-prepare.",
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

    parser.add_argument(
        "--ollama",
        action="store_true",
        help="После очистки транскрипта обработать чанки локальной моделью через Ollama.",
    )

    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Базовый URL Ollama API.",
    )

    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen2.5:7b",
        help="Название локальной модели Ollama, например qwen2.5:7b, llama3.1:8b, gemma3.",
    )

    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=900,
        help="Таймаут одного запроса к Ollama в секундах.",
    )

    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=0.1,
        help="Температура генерации Ollama. Для методической обработки лучше 0.0–0.2.",
    )

    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=8192,
        help="Контекст Ollama. Увеличьте, если модель и железо позволяют.",
    )

    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=3000,
        help="Максимальное количество генерируемых токенов на один ответ Ollama.",
    )

    parser.add_argument(
        "--ollama-keep-alive",
        type=str,
        default="10m",
        help="Сколько держать модель Ollama загруженной после запроса.",
    )

    parser.add_argument(
        "--no-ollama-final-synthesis",
        action="store_true",
        help="Не собирать итоговый протокол; обработать только отдельные чанки.",
    )

    parser.add_argument(
        "--ollama-final-max-chars",
        type=int,
        default=45000,
        help="Максимальный размер финального prompt для сборки общего протокола.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    input_dir = Path(args.input_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        logging.error("Папка не найдена: %s", input_dir)
        raise SystemExit(1)

    if not args.skip_audio_prepare:
        ensure_ffmpeg_tools()
    elif not has_ffmpeg_tools():
        logging.warning(
            "ffmpeg/ffprobe не найдены. Это допустимо при --skip-audio-prepare, "
            "так как faster-whisper использует PyAV, но длительность до транскрибации может не логироваться."
        )

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

    whisper_config = FasterWhisperConfig(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
        beam_size=args.beam_size,
        language=args.language,
        initial_prompt=initial_prompt,
        condition_on_previous_text=args.condition_on_previous_text,
        vad_filter=args.vad_filter,
        vad_min_silence_duration_ms=args.vad_min_silence_duration_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        word_timestamps=args.word_timestamps,
    )

    ollama_config = None

    if args.ollama:
        ollama_config = OllamaConfig(
            enabled=True,
            base_url=args.ollama_url,
            model=args.ollama_model,
            timeout=args.ollama_timeout,
            temperature=args.ollama_temperature,
            num_ctx=args.ollama_num_ctx,
            num_predict=args.ollama_num_predict,
            keep_alive=args.ollama_keep_alive,
            final_synthesis=not args.no_ollama_final_synthesis,
            final_max_chars=args.ollama_final_max_chars,
        )

    logging.info("Найдено аудиофайлов: %s", len(audio_files))
    logging.info("CPU логических ядер: %s", os.cpu_count())
    logging.info("OMP_NUM_THREADS: %s", os.environ.get("OMP_NUM_THREADS", "не задано"))

    try:
        model = load_faster_whisper_model(whisper_config)
    except Exception as exc:
        logging.exception("Не удалось загрузить faster-whisper модель: %s", exc)
        raise SystemExit(1) from exc

    logging.info("Модель загружена.")
    logging.info("")

    success_count = 0
    failed_files: list[str] = []

    for audio_file in audio_files:
        try:
            process_one_audio_file(
                audio_file=audio_file,
                model=model,
                whisper_config=whisper_config,
                clean_mode=args.clean_mode,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                prompt_include_text_limit=args.prompt_include_text_limit,
                keep_prepared_wav=args.keep_prepared_wav,
                skip_audio_prepare=args.skip_audio_prepare,
                audio_filter=args.audio_filter,
                ollama_config=ollama_config,
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
