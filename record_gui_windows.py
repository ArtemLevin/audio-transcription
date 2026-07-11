from __future__ import annotations

import datetime as dt
import logging
import queue
import re
import threading
import time
import tkinter as tk
import traceback
import wave
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import messagebox
from tkinter import ttk

import numpy as np
import pyaudiowpatch as pyaudio


# =========================
# ГЛОБАЛЬНЫЕ НАСТРОЙКИ
# =========================

FORMAT = pyaudio.paInt16
DEFAULT_RATE = 48000
DEFAULT_CHUNK = 256

RECORDS_DIR = Path("records")
LOGS_DIR = Path("logs")

LOGGER_NAME = "audio_recorder"
logger = logging.getLogger(LOGGER_NAME)


# =========================
# ЛОГИРОВАНИЕ
# =========================

def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def configure_logging() -> Path:
    """
    Настраивает подробное логирование:
    - DEBUG и выше — в файл;
    - INFO и выше — в консоль PowerShell.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOGS_DIR / f"audio_recorder_{timestamp()}.log"

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(threadName)s | "
            "%(name)s | %(funcName)s:%(lineno)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Логирование настроено.")
    logger.info("Файл лога: %s", log_path.resolve())

    return log_path


def log_full_exception(message: str) -> None:
    """
    Логирует полную информацию об исключении.
    """
    logger.error(message)
    logger.error("Traceback:\n%s", traceback.format_exc())


def device_to_short_text(device: dict) -> str:
    """
    Человекочитаемое описание аудиоустройства.
    """
    return (
        f"index={device.get('index')}, "
        f"name={device.get('name')!r}, "
        f"maxInputChannels={device.get('maxInputChannels')}, "
        f"maxOutputChannels={device.get('maxOutputChannels')}, "
        f"defaultSampleRate={device.get('defaultSampleRate')}, "
        f"isLoopbackDevice={device.get('isLoopbackDevice')}, "
        f"hostApi={device.get('hostApi')}"
    )


def log_audio_stats(label: str, audio: np.ndarray) -> None:
    """
    Пишет в лог простую диагностику аудиосигнала.
    Полезно для отладки тишины, перегруза и выбора неправильного устройства.
    """
    if audio.size == 0:
        logger.debug("%s stats: empty audio array", label)
        return

    audio_f = audio.astype(np.float32)

    peak = float(np.max(np.abs(audio_f)))
    rms = float(np.sqrt(np.mean(audio_f ** 2)))
    min_value = int(np.min(audio))
    max_value = int(np.max(audio))

    logger.debug(
        "%s stats: shape=%s, dtype=%s, peak=%.2f, rms=%.2f, min=%s, max=%s",
        label,
        audio.shape,
        audio.dtype,
        peak,
        rms,
        min_value,
        max_value,
    )


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def sanitize_filename(value: str) -> str:
    """
    Убирает символы, запрещённые в именах файлов Windows.
    Кириллицу оставляем.
    """
    logger.debug("sanitize_filename input=%r", value)

    value = value.strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", " ", value)

    result = value or "Без имени"

    logger.debug("sanitize_filename output=%r", result)

    return result


def bytes_to_stereo_int16(data: bytes, channels: int) -> np.ndarray:
    """
    Преобразует байты из PyAudio в stereo int16 numpy-массив:
    shape = frames x 2.
    """
    logger.debug(
        "bytes_to_stereo_int16 called: len(data)=%s, channels=%s",
        len(data),
        channels,
    )

    audio = np.frombuffer(data, dtype=np.int16)

    if len(audio) == 0:
        logger.warning("bytes_to_stereo_int16: получен пустой аудиобуфер.")
        return np.zeros((0, 2), dtype=np.int16)

    usable_len = len(audio) - (len(audio) % channels)

    if usable_len <= 0:
        logger.warning(
            "bytes_to_stereo_int16: usable_len <= 0. len(audio)=%s, channels=%s",
            len(audio),
            channels,
        )
        return np.zeros((0, 2), dtype=np.int16)

    if usable_len != len(audio):
        logger.debug(
            "bytes_to_stereo_int16: обрезаем буфер с %s до %s значений.",
            len(audio),
            usable_len,
        )

    audio = audio[:usable_len].reshape(-1, channels)

    if channels == 1:
        audio = np.repeat(audio, 2, axis=1)
        logger.debug("bytes_to_stereo_int16: mono -> stereo.")
    elif channels >= 2:
        audio = audio[:, :2]
        logger.debug("bytes_to_stereo_int16: multi-channel -> first 2 channels.")

    result = audio.astype(np.int16, copy=False)

    logger.debug("bytes_to_stereo_int16 result shape=%s", result.shape)

    return result


def channels_for_recording(device: dict) -> int:
    """
    Берём 1 или 2 входных канала.
    Для итоговых WAV всё равно приводим к стерео.
    """
    logger.debug("channels_for_recording device: %s", device_to_short_text(device))

    channels = int(device.get("maxInputChannels", 0))

    if channels <= 0:
        logger.error("Устройство не умеет записывать: %s", device_to_short_text(device))
        raise RuntimeError(f"Устройство не умеет записывать: {device['name']}")

    result = min(channels, 2)

    logger.info(
        "Для записи выбрано каналов: %s для устройства %r",
        result,
        device.get("name"),
    )

    return result


def create_wav_writer(path: Path, rate: int) -> wave.Wave_write:
    """
    Создаёт WAV writer для stereo 16-bit PCM.
    """
    logger.info("Создаём WAV-файл: %s", path.resolve())
    logger.debug("WAV params: channels=2, sampwidth=2, framerate=%s", rate)

    path.parent.mkdir(parents=True, exist_ok=True)

    wf = wave.open(str(path), "wb")
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(rate)

    logger.debug("WAV writer создан успешно: %s", path.resolve())

    return wf


def mix_int16(
        mic_audio: np.ndarray,
        system_audio: np.ndarray,
        mic_gain: float = 1.0,
        system_gain: float = 1.0,
) -> np.ndarray:
    """
    Смешивает микрофон и системный звук с указанными коэффициентами громкости.
    """
    logger.debug(
        "mix_int16 called: mic_shape=%s, system_shape=%s, mic_gain=%s, system_gain=%s",
        mic_audio.shape,
        system_audio.shape,
        mic_gain,
        system_gain,
    )

    mixed = (
            mic_audio.astype(np.float32) * mic_gain
            + system_audio.astype(np.float32) * system_gain
    )

    clipped = np.clip(mixed, -32768, 32767).astype(np.int16)

    logger.debug("mix_int16 result shape=%s", clipped.shape)

    return clipped


def format_time(seconds: float) -> str:
    seconds_int = int(seconds)
    minutes = seconds_int // 60
    rest = seconds_int % 60
    return f"{minutes}:{rest:02d}"


# =========================
# КЛАСС ЗАПИСИ
# =========================

class AudioRecorder:
    def __init__(
            self,
            *,
            mic_device_index: int,
            loopback_device_index: int,
            student_name: str,
            duration_minutes: float,
            rate: int,
            chunk: int,
            mic_gain: float,
            system_gain: float,
            stop_event: threading.Event,
            ui_queue: queue.Queue,
    ) -> None:
        logger.debug("AudioRecorder.__init__ called.")

        self.mic_device_index = mic_device_index
        self.loopback_device_index = loopback_device_index
        self.student_name = sanitize_filename(student_name)
        self.duration_minutes = duration_minutes
        self.rate = rate
        self.chunk = chunk
        self.mic_gain = mic_gain
        self.system_gain = system_gain
        self.stop_event = stop_event
        self.ui_queue = ui_queue

        logger.info(
            "AudioRecorder создан: student=%r, duration_minutes=%s, rate=%s, "
            "chunk=%s, mic_index=%s, loopback_index=%s, mic_gain=%s, system_gain=%s",
            self.student_name,
            self.duration_minutes,
            self.rate,
            self.chunk,
            self.mic_device_index,
            self.loopback_device_index,
            self.mic_gain,
            self.system_gain,
        )

    def run(self) -> None:
        logger.info("Поток записи запущен.")

        p = None
        mic_stream = None
        system_stream = None

        mic_wav = None
        system_wav = None
        mix_wav = None

        try:
            logger.debug("Инициализация PyAudio.")
            p = pyaudio.PyAudio()
            logger.info("PyAudio инициализирован.")

            logger.debug("Получаем информацию о микрофоне index=%s.", self.mic_device_index)
            mic_device = p.get_device_info_by_index(self.mic_device_index)
            logger.info("Микрофон: %s", device_to_short_text(mic_device))

            logger.debug(
                "Получаем информацию о loopback index=%s.",
                self.loopback_device_index,
            )
            loopback_device = p.get_device_info_by_index(self.loopback_device_index)
            logger.info("Loopback: %s", device_to_short_text(loopback_device))

            mic_channels = channels_for_recording(mic_device)
            system_channels = channels_for_recording(loopback_device)

            total_seconds = self.duration_minutes * 60
            frames_total = int(total_seconds * self.rate)
            frames_done = 0

            logger.info(
                "Расчёт длительности: total_seconds=%s, frames_total=%s",
                total_seconds,
                frames_total,
            )

            session_timestamp = timestamp()
            session_dir = RECORDS_DIR / self.student_name / session_timestamp
            base_name = f"{self.student_name}_{session_timestamp}"

            mic_path = session_dir / f"{base_name}_mic.wav"
            system_path = session_dir / f"{base_name}_system.wav"
            mix_path = session_dir / f"{base_name}_mix.wav"

            logger.info("Папка сессии: %s", session_dir.resolve())
            logger.info("Файл микрофона: %s", mic_path.resolve())
            logger.info("Файл системного звука: %s", system_path.resolve())
            logger.info("Файл микса: %s", mix_path.resolve())

            mic_wav = create_wav_writer(mic_path, self.rate)
            system_wav = create_wav_writer(system_path, self.rate)
            mix_wav = create_wav_writer(mix_path, self.rate)

            self.ui_queue.put(
                {
                    "type": "status",
                    "text": (
                        "Запись началась.\n"
                        f"Ученик: {self.student_name}\n"
                        f"Микрофон: {mic_device['name']}\n"
                        f"Звук компьютера: {loopback_device['name']}"
                    ),
                }
            )

            logger.info("Открываем поток микрофона.")
            logger.debug(
                "mic_stream params: format=%s, channels=%s, rate=%s, "
                "input_device_index=%s, frames_per_buffer=%s",
                FORMAT,
                mic_channels,
                self.rate,
                self.mic_device_index,
                self.chunk,
            )

            mic_stream = p.open(
                format=FORMAT,
                channels=mic_channels,
                rate=self.rate,
                input=True,
                input_device_index=self.mic_device_index,
                frames_per_buffer=self.chunk,
            )

            logger.info("Поток микрофона открыт.")

            logger.info("Открываем loopback-поток.")
            logger.debug(
                "system_stream params: format=%s, channels=%s, rate=%s, "
                "input_device_index=%s, frames_per_buffer=%s",
                FORMAT,
                system_channels,
                self.rate,
                self.loopback_device_index,
                self.chunk,
            )

            system_stream = p.open(
                format=FORMAT,
                channels=system_channels,
                rate=self.rate,
                input=True,
                input_device_index=self.loopback_device_index,
                frames_per_buffer=self.chunk,
            )

            logger.info("Loopback-поток открыт.")
            logger.info("Основной цикл записи начался.")

            last_progress_time = 0.0
            last_debug_audio_stats_time = 0.0
            last_info_progress_time = 0.0
            iteration = 0

            while frames_done < frames_total and not self.stop_event.is_set():
                iteration += 1
                frames_now = min(self.chunk, frames_total - frames_done)

                logger.debug(
                    "Итерация записи #%s: frames_now=%s, frames_done=%s/%s",
                    iteration,
                    frames_now,
                    frames_done,
                    frames_total,
                )

                mic_data = mic_stream.read(
                    frames_now,
                    exception_on_overflow=False,
                )

                system_data = system_stream.read(
                    frames_now,
                    exception_on_overflow=False,
                )

                logger.debug(
                    "Прочитаны байты: mic=%s bytes, system=%s bytes",
                    len(mic_data),
                    len(system_data),
                )

                mic_audio = bytes_to_stereo_int16(mic_data, mic_channels)
                system_audio = bytes_to_stereo_int16(system_data, system_channels)

                n = min(len(mic_audio), len(system_audio))

                if n <= 0:
                    logger.warning(
                        "После преобразования нет аудиокадров. "
                        "mic_len=%s, system_len=%s",
                        len(mic_audio),
                        len(system_audio),
                    )
                    continue

                if len(mic_audio) != len(system_audio):
                    logger.debug(
                        "Длины аудио различаются. mic_len=%s, system_len=%s, берём n=%s",
                        len(mic_audio),
                        len(system_audio),
                        n,
                    )

                mic_audio = mic_audio[:n]
                system_audio = system_audio[:n]

                mixed_audio = mix_int16(
                    mic_audio,
                    system_audio,
                    mic_gain=self.mic_gain,
                    system_gain=self.system_gain,
                )

                mic_wav.writeframes(mic_audio.tobytes())
                system_wav.writeframes(system_audio.tobytes())
                mix_wav.writeframes(mixed_audio.tobytes())

                frames_done += n

                now = time.time()
                elapsed = frames_done / self.rate
                progress = min(100.0, elapsed / total_seconds * 100.0)

                if now - last_progress_time >= 0.2:
                    self.ui_queue.put(
                        {
                            "type": "progress",
                            "elapsed": elapsed,
                            "total": total_seconds,
                            "progress": progress,
                        }
                    )
                    last_progress_time = now

                if now - last_info_progress_time >= 5.0:
                    logger.info(
                        "Прогресс записи: elapsed=%s/%s, progress=%.2f%%, frames_done=%s/%s",
                        format_time(elapsed),
                        format_time(total_seconds),
                        progress,
                        frames_done,
                        frames_total,
                    )
                    last_info_progress_time = now

                if now - last_debug_audio_stats_time >= 5.0:
                    log_audio_stats("MIC", mic_audio)
                    log_audio_stats("SYSTEM", system_audio)
                    log_audio_stats("MIX", mixed_audio)
                    last_debug_audio_stats_time = now

            elapsed = frames_done / self.rate

            logger.info(
                "Основной цикл записи завершён. frames_done=%s, elapsed=%s сек.",
                frames_done,
                elapsed,
            )

            if self.stop_event.is_set():
                finish_text = "Запись остановлена пользователем."
                logger.info("Причина завершения: stop_event установлен.")
            else:
                finish_text = "Запись завершена по заданной длительности."
                logger.info("Причина завершения: достигнута заданная длительность.")

            self.ui_queue.put(
                {
                    "type": "done",
                    "text": finish_text,
                    "elapsed": elapsed,
                    "files": {
                        "mic": str(mic_path),
                        "system": str(system_path),
                        "mix": str(mix_path),
                    },
                }
            )

            logger.info("Событие done отправлено в GUI.")

        except Exception:
            log_full_exception("Ошибка в AudioRecorder.run().")

            self.ui_queue.put(
                {
                    "type": "error",
                    "text": traceback.format_exc(),
                }
            )

        finally:
            logger.info("AudioRecorder.run finally: начинаем закрытие ресурсов.")

            self.stop_event.set()

            if mic_stream is not None:
                try:
                    logger.debug("Останавливаем mic_stream.")
                    mic_stream.stop_stream()
                    logger.debug("mic_stream остановлен.")
                except Exception:
                    log_full_exception("Ошибка при остановке mic_stream.")

                try:
                    logger.debug("Закрываем mic_stream.")
                    mic_stream.close()
                    logger.debug("mic_stream закрыт.")
                except Exception:
                    log_full_exception("Ошибка при закрытии mic_stream.")

            if system_stream is not None:
                try:
                    logger.debug("Останавливаем system_stream.")
                    system_stream.stop_stream()
                    logger.debug("system_stream остановлен.")
                except Exception:
                    log_full_exception("Ошибка при остановке system_stream.")

                try:
                    logger.debug("Закрываем system_stream.")
                    system_stream.close()
                    logger.debug("system_stream закрыт.")
                except Exception:
                    log_full_exception("Ошибка при закрытии system_stream.")

            for name, wav_file in (
                    ("mic_wav", mic_wav),
                    ("system_wav", system_wav),
                    ("mix_wav", mix_wav),
            ):
                if wav_file is not None:
                    try:
                        logger.debug("Закрываем WAV writer: %s.", name)
                        wav_file.close()
                        logger.debug("WAV writer закрыт: %s.", name)
                    except Exception:
                        log_full_exception(f"Ошибка при закрытии WAV writer: {name}.")

            if p is not None:
                try:
                    logger.debug("Завершаем PyAudio.")
                    p.terminate()
                    logger.info("PyAudio завершён.")
                except Exception:
                    log_full_exception("Ошибка при p.terminate().")

            logger.info("Поток записи завершён полностью.")


# =========================
# GUI
# =========================

class RecorderApp(tk.Tk):
    def __init__(self, log_path: Path) -> None:
        logger.info("Инициализация RecorderApp.")

        super().__init__()

        self.log_path = log_path

        self.title("Запись микрофона и звука компьютера")
        self.geometry("900x620")
        self.minsize(820, 560)

        self.ui_queue: queue.Queue = queue.Queue()
        self.stop_event: threading.Event | None = None
        self.recording_thread: threading.Thread | None = None

        self.mic_devices: dict[str, dict] = {}
        self.loopback_devices: dict[str, dict] = {}

        self.student_var = tk.StringVar()
        self.duration_var = tk.StringVar(value="60")
        self.rate_var = tk.StringVar(value=str(DEFAULT_RATE))
        self.chunk_var = tk.StringVar(value=str(DEFAULT_CHUNK))
        self.mic_gain_var = tk.StringVar(value="1.0")
        self.system_gain_var = tk.StringVar(value="1.0")

        self.mic_device_var = tk.StringVar()
        self.loopback_device_var = tk.StringVar()

        self.status_var = tk.StringVar(value="Готово к записи.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.time_var = tk.StringVar(value="0:00 / 0:00")

        self._build_ui()
        self.refresh_devices()
        self.after(100, self.process_ui_queue)

        logger.info("RecorderApp инициализирован.")

    def _build_ui(self) -> None:
        logger.debug("Строим GUI.")

        main = ttk.Frame(self, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main,
            text="Запись занятия: микрофон + звук компьютера",
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(anchor=tk.W, pady=(0, 16))

        form = ttk.LabelFrame(main, text="Параметры записи", padding=12)
        form.pack(fill=tk.X)

        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Имя ученика:").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, 8),
            pady=6,
        )

        ttk.Entry(form, textvariable=self.student_var).grid(
            row=0,
            column=1,
            sticky=tk.EW,
            pady=6,
        )

        ttk.Label(form, text="Длительность, минут:").grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=(0, 8),
            pady=6,
        )

        ttk.Entry(form, textvariable=self.duration_var, width=12).grid(
            row=1,
            column=1,
            sticky=tk.W,
            pady=6,
        )

        ttk.Label(form, text="Микрофон:").grid(
            row=2,
            column=0,
            sticky=tk.W,
            padx=(0, 8),
            pady=6,
        )

        self.mic_combo = ttk.Combobox(
            form,
            textvariable=self.mic_device_var,
            state="readonly",
        )
        self.mic_combo.grid(row=2, column=1, sticky=tk.EW, pady=6)

        ttk.Label(form, text="Loopback / звук компьютера:").grid(
            row=3,
            column=0,
            sticky=tk.W,
            padx=(0, 8),
            pady=6,
        )

        self.loopback_combo = ttk.Combobox(
            form,
            textvariable=self.loopback_device_var,
            state="readonly",
        )
        self.loopback_combo.grid(row=3, column=1, sticky=tk.EW, pady=6)

        advanced = ttk.LabelFrame(main, text="Дополнительно", padding=12)
        advanced.pack(fill=tk.X, pady=(12, 0))

        for col in range(4):
            advanced.columnconfigure(col, weight=1)

        ttk.Label(advanced, text="Частота:").grid(
            row=0,
            column=0,
            sticky=tk.W,
            pady=6,
        )

        ttk.Entry(advanced, textvariable=self.rate_var, width=12).grid(
            row=0,
            column=1,
            sticky=tk.W,
            pady=6,
        )

        ttk.Label(advanced, text="Размер блока:").grid(
            row=0,
            column=2,
            sticky=tk.W,
            pady=6,
        )

        ttk.Entry(advanced, textvariable=self.chunk_var, width=12).grid(
            row=0,
            column=3,
            sticky=tk.W,
            pady=6,
        )

        ttk.Label(advanced, text="Громкость микрофона:").grid(
            row=1,
            column=0,
            sticky=tk.W,
            pady=6,
        )

        ttk.Entry(advanced, textvariable=self.mic_gain_var, width=12).grid(
            row=1,
            column=1,
            sticky=tk.W,
            pady=6,
        )

        ttk.Label(advanced, text="Громкость компьютера:").grid(
            row=1,
            column=2,
            sticky=tk.W,
            pady=6,
        )

        ttk.Entry(advanced, textvariable=self.system_gain_var, width=12).grid(
            row=1,
            column=3,
            sticky=tk.W,
            pady=6,
        )

        buttons = ttk.Frame(main)
        buttons.pack(fill=tk.X, pady=16)

        self.refresh_button = ttk.Button(
            buttons,
            text="Обновить устройства",
            command=self.refresh_devices,
        )
        self.refresh_button.pack(side=tk.LEFT)

        self.start_button = ttk.Button(
            buttons,
            text="Старт записи",
            command=self.start_recording,
        )
        self.start_button.pack(side=tk.LEFT, padx=(12, 0))

        self.stop_button = ttk.Button(
            buttons,
            text="Стоп",
            command=self.stop_recording,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(12, 0))

        progress_box = ttk.LabelFrame(main, text="Состояние", padding=12)
        progress_box.pack(fill=tk.BOTH, expand=True)

        self.progress_bar = ttk.Progressbar(
            progress_box,
            variable=self.progress_var,
            maximum=100,
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(progress_box, textvariable=self.time_var).pack(anchor=tk.W)

        self.status_text = tk.Text(
            progress_box,
            height=12,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.write_status("Готово. Укажите ученика, длительность и устройства.")
        self.write_status(f"Файл лога: {self.log_path.resolve()}")

        logger.debug("GUI построен.")

    def write_status(self, text: str) -> None:
        logger.info("GUI STATUS: %s", text.replace("\n", " | "))

        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, text + "\n")
        self.status_text.see(tk.END)
        self.status_text.configure(state=tk.DISABLED)
        self.status_var.set(text)

    def refresh_devices(self) -> None:
        logger.info("Обновление списка устройств запрошено.")

        if self.is_recording():
            logger.warning("Попытка обновить устройства во время записи.")
            messagebox.showwarning(
                "Запись идёт",
                "Нельзя обновлять устройства во время записи.",
            )
            return

        self.mic_devices.clear()
        self.loopback_devices.clear()

        p = None

        try:
            logger.debug("Инициализация PyAudio для списка устройств.")
            p = pyaudio.PyAudio()

            logger.info("Начинаем перечисление аудиоустройств.")

            for device in p.get_device_info_generator():
                logger.debug("Обнаружено устройство: %s", device_to_short_text(device))

                name = device["name"]
                index = int(device["index"])
                max_input = int(device.get("maxInputChannels", 0))
                is_loopback = bool(device.get("isLoopbackDevice"))

                if max_input <= 0:
                    logger.debug(
                        "Пропускаем устройство без input-каналов: index=%s, name=%r",
                        index,
                        name,
                    )
                    continue

                label = f"[{index}] {name}"

                if is_loopback:
                    logger.info("Добавлено loopback-устройство: %s", label)
                    self.loopback_devices[label] = device
                else:
                    logger.info("Добавлен микрофон: %s", label)
                    self.mic_devices[label] = device

        except Exception:
            log_full_exception("Ошибка при обновлении списка устройств.")
            messagebox.showerror(
                "Ошибка устройств",
                "Не удалось получить список аудиоустройств. Подробности в лог-файле.",
            )
            return

        finally:
            if p is not None:
                try:
                    logger.debug("Завершаем PyAudio после refresh_devices.")
                    p.terminate()
                except Exception:
                    log_full_exception("Ошибка при p.terminate() в refresh_devices.")

        mic_values = list(self.mic_devices.keys())
        loopback_values = list(self.loopback_devices.keys())

        logger.info(
            "Итог обновления устройств: microphones=%s, loopbacks=%s",
            len(mic_values),
            len(loopback_values),
        )

        self.mic_combo["values"] = mic_values
        self.loopback_combo["values"] = loopback_values

        self._select_preferred_device(
            combo=self.mic_combo,
            var=self.mic_device_var,
            values=mic_values,
            preferred_parts=["fifine Chat", "fifine", "Microphone"],
        )

        self._select_preferred_device(
            combo=self.loopback_combo,
            var=self.loopback_device_var,
            values=loopback_values,
            preferred_parts=["G733", "Динамики", "Headphones", "Speakers"],
        )

        self.write_status(
            f"Устройства обновлены. "
            f"Микрофонов: {len(mic_values)}. "
            f"Loopback-устройств: {len(loopback_values)}."
        )

        logger.info("Выбран микрофон по умолчанию: %r", self.mic_device_var.get())
        logger.info("Выбран loopback по умолчанию: %r", self.loopback_device_var.get())

    def _select_preferred_device(
            self,
            *,
            combo: ttk.Combobox,
            var: tk.StringVar,
            values: list[str],
            preferred_parts: list[str],
    ) -> None:
        logger.debug(
            "_select_preferred_device values=%s, preferred_parts=%s",
            values,
            preferred_parts,
        )

        if not values:
            logger.warning("Нет доступных значений для Combobox.")
            var.set("")
            return

        for part in preferred_parts:
            part_lower = part.lower()

            for value in values:
                if part_lower in value.lower():
                    logger.info(
                        "Автовыбор устройства: part=%r, selected=%r",
                        part,
                        value,
                    )
                    var.set(value)
                    return

        logger.info("Автовыбор fallback: selected=%r", values[0])
        var.set(values[0])

    def validate_inputs(self) -> tuple[str, float, int, int, float, float]:
        logger.info("Начинаем валидацию параметров GUI.")

        student = self.student_var.get().strip()
        logger.debug("student raw=%r", student)

        if not student:
            logger.warning("Валидация не пройдена: пустое имя ученика.")
            raise ValueError("Укажите имя ученика.")

        try:
            duration_minutes = float(self.duration_var.get().replace(",", "."))
            logger.debug("duration_minutes=%s", duration_minutes)
        except ValueError as exc:
            logger.warning("Валидация не пройдена: некорректная длительность.")
            raise ValueError("Длительность должна быть числом.") from exc

        if duration_minutes <= 0:
            logger.warning("Валидация не пройдена: duration_minutes <= 0.")
            raise ValueError("Длительность должна быть больше 0 минут.")

        try:
            rate = int(self.rate_var.get())
            logger.debug("rate=%s", rate)
        except ValueError as exc:
            logger.warning("Валидация не пройдена: частота не int.")
            raise ValueError("Частота должна быть целым числом.") from exc

        if rate <= 0:
            logger.warning("Валидация не пройдена: rate <= 0.")
            raise ValueError("Частота должна быть больше 0.")

        try:
            chunk = int(self.chunk_var.get())
            logger.debug("chunk=%s", chunk)
        except ValueError as exc:
            logger.warning("Валидация не пройдена: chunk не int.")
            raise ValueError("Размер блока должен быть целым числом.") from exc

        if chunk <= 0:
            logger.warning("Валидация не пройдена: chunk <= 0.")
            raise ValueError("Размер блока должен быть больше 0.")

        try:
            mic_gain = float(self.mic_gain_var.get().replace(",", "."))
            logger.debug("mic_gain=%s", mic_gain)
        except ValueError as exc:
            logger.warning("Валидация не пройдена: mic_gain не float.")
            raise ValueError("Громкость микрофона должна быть числом.") from exc

        try:
            system_gain = float(self.system_gain_var.get().replace(",", "."))
            logger.debug("system_gain=%s", system_gain)
        except ValueError as exc:
            logger.warning("Валидация не пройдена: system_gain не float.")
            raise ValueError("Громкость компьютера должна быть числом.") from exc

        if mic_gain < 0 or system_gain < 0:
            logger.warning("Валидация не пройдена: отрицательная громкость.")
            raise ValueError("Громкость не может быть отрицательной.")

        if not self.mic_device_var.get():
            logger.warning("Валидация не пройдена: микрофон не выбран.")
            raise ValueError("Выберите микрофон.")

        if not self.loopback_device_var.get():
            logger.warning("Валидация не пройдена: loopback не выбран.")
            raise ValueError("Выберите loopback-устройство.")

        logger.info(
            "Валидация успешна: student=%r, duration=%s, rate=%s, chunk=%s, "
            "mic_gain=%s, system_gain=%s, mic=%r, loopback=%r",
            student,
            duration_minutes,
            rate,
            chunk,
            mic_gain,
            system_gain,
            self.mic_device_var.get(),
            self.loopback_device_var.get(),
        )

        return student, duration_minutes, rate, chunk, mic_gain, system_gain

    def start_recording(self) -> None:
        logger.info("Нажата кнопка Старт записи.")

        if self.is_recording():
            logger.warning("Попытка стартовать запись, когда запись уже идёт.")
            messagebox.showwarning(
                "Запись уже идёт",
                "Сначала остановите текущую запись.",
            )
            return

        try:
            student, duration_minutes, rate, chunk, mic_gain, system_gain = (
                self.validate_inputs()
            )
        except ValueError as exc:
            logger.warning("Ошибка параметров при старте: %s", exc)
            messagebox.showerror("Ошибка параметров", str(exc))
            return

        mic_label = self.mic_device_var.get()
        loopback_label = self.loopback_device_var.get()

        logger.info("Выбранный mic_label=%r", mic_label)
        logger.info("Выбранный loopback_label=%r", loopback_label)

        mic_device = self.mic_devices.get(mic_label)
        loopback_device = self.loopback_devices.get(loopback_label)

        if mic_device is None:
            logger.error("Выбранный микрофон не найден в self.mic_devices.")
            messagebox.showerror("Ошибка", "Выбранный микрофон не найден.")
            return

        if loopback_device is None:
            logger.error("Выбранное loopback-устройство не найдено.")
            messagebox.showerror("Ошибка", "Выбранное loopback-устройство не найдено.")
            return

        logger.info("Стартуем запись с микрофоном: %s", device_to_short_text(mic_device))
        logger.info("Стартуем запись с loopback: %s", device_to_short_text(loopback_device))

        self.stop_event = threading.Event()
        self.progress_var.set(0)
        self.time_var.set("0:00 / 0:00")

        recorder = AudioRecorder(
            mic_device_index=int(mic_device["index"]),
            loopback_device_index=int(loopback_device["index"]),
            student_name=student,
            duration_minutes=duration_minutes,
            rate=rate,
            chunk=chunk,
            mic_gain=mic_gain,
            system_gain=system_gain,
            stop_event=self.stop_event,
            ui_queue=self.ui_queue,
        )

        self.recording_thread = threading.Thread(
            target=recorder.run,
            daemon=True,
            name="AudioRecorderThread",
        )

        self.set_controls_recording_state(is_recording=True)
        self.write_status("Подготовка к записи...")

        logger.info("Запускаем поток AudioRecorderThread.")
        self.recording_thread.start()

    def stop_recording(self) -> None:
        logger.info("Нажата кнопка Стоп.")

        if self.stop_event is not None:
            logger.info("Устанавливаем stop_event.")
            self.stop_event.set()
            self.write_status("Останавливаю запись. Файлы будут сохранены.")
        else:
            logger.warning("stop_recording вызван, но stop_event is None.")

    def is_recording(self) -> bool:
        result = (
                self.recording_thread is not None
                and self.recording_thread.is_alive()
        )

        logger.debug("is_recording -> %s", result)

        return result

    def set_controls_recording_state(self, *, is_recording: bool) -> None:
        logger.debug("set_controls_recording_state is_recording=%s", is_recording)

        state_when_idle = tk.DISABLED if is_recording else tk.NORMAL
        combo_state_when_idle = tk.DISABLED if is_recording else "readonly"

        self.start_button.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if is_recording else tk.DISABLED)
        self.refresh_button.configure(state=state_when_idle)

        self.mic_combo.configure(state=combo_state_when_idle)
        self.loopback_combo.configure(state=combo_state_when_idle)

    def process_ui_queue(self) -> None:
        try:
            while True:
                event = self.ui_queue.get_nowait()
                event_type = event.get("type")

                logger.debug("Получено событие из ui_queue: %s", event_type)

                if event_type == "status":
                    self.write_status(event["text"])

                elif event_type == "progress":
                    elapsed = float(event["elapsed"])
                    total = float(event["total"])
                    progress = float(event["progress"])

                    self.progress_var.set(progress)
                    self.time_var.set(
                        f"{format_time(elapsed)} / {format_time(total)}"
                    )

                elif event_type == "done":
                    logger.info("GUI получил событие done.")

                    self.progress_var.set(100)
                    elapsed = float(event["elapsed"])

                    self.write_status(event["text"])
                    self.write_status(f"Фактическая длительность: {format_time(elapsed)}")

                    files = event["files"]

                    self.write_status("Файлы сохранены:")
                    self.write_status(f"Микрофон: {files['mic']}")
                    self.write_status(f"Звук компьютера: {files['system']}")
                    self.write_status(f"Общий микс: {files['mix']}")

                    logger.info("Saved mic file: %s", files["mic"])
                    logger.info("Saved system file: %s", files["system"])
                    logger.info("Saved mix file: %s", files["mix"])

                    self.set_controls_recording_state(is_recording=False)

                elif event_type == "error":
                    logger.error("GUI получил событие error.")

                    self.write_status("Ошибка записи:")
                    self.write_status(event["text"])

                    messagebox.showerror(
                        "Ошибка записи",
                        "Произошла ошибка. Полная информация записана в лог-файл.",
                    )

                    self.set_controls_recording_state(is_recording=False)

                else:
                    logger.warning("Неизвестный тип события ui_queue: %r", event_type)

        except queue.Empty:
            pass

        if not self.is_recording() and self.stop_button["state"] == tk.NORMAL:
            logger.debug("Поток записи не активен, но кнопка Стоп включена. Сбрасываем UI.")
            self.set_controls_recording_state(is_recording=False)

        self.after(100, self.process_ui_queue)

    def on_close(self) -> None:
        logger.info("Запрошено закрытие окна.")

        if self.is_recording():
            logger.warning("Закрытие окна во время записи.")

            answer = messagebox.askyesno(
                "Запись идёт",
                "Запись ещё идёт. Остановить запись и закрыть программу?",
            )

            logger.info("Ответ пользователя на закрытие во время записи: %s", answer)

            if not answer:
                logger.info("Пользователь отменил закрытие.")
                return

            if self.stop_event is not None:
                logger.info("Устанавливаем stop_event перед закрытием.")
                self.stop_event.set()

            self.write_status("Остановка перед закрытием...")

            self.after(500, self.destroy)
            return

        logger.info("Окно закрывается.")
        self.destroy()


# =========================
# ТОЧКА ВХОДА
# =========================

def main() -> None:
    log_path = configure_logging()

    logger.info("Приложение запускается.")
    logger.info("Рабочая директория: %s", Path.cwd().resolve())
    logger.info("Папка записей: %s", RECORDS_DIR.resolve())
    logger.info("Папка логов: %s", LOGS_DIR.resolve())

    try:
        app = RecorderApp(log_path=log_path)
        app.protocol("WM_DELETE_WINDOW", app.on_close)
        logger.info("Запускаем Tkinter mainloop.")
        app.mainloop()
        logger.info("Tkinter mainloop завершён.")
    except Exception:
        log_full_exception("Критическая ошибка в main().")
        raise
    finally:
        logger.info("Приложение завершено.")


if __name__ == "__main__":
    main()