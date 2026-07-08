Audio Lesson Transcriber Faster
Скрипт для обработки аудиозаписей учебных занятий:
```text
аудиофайл
  → транскрипт через faster-whisper
  → очищенный транскрипт
  → смысловая версия для будущего пособия
  → чанки для LLM
  → опциональный учебный протокол через Ollama
```
Скрипт ориентирован на русскоязычные занятия по математике, физике и химии. Основной сценарий: получить качественную основу для последующей генерации LaTeX-пособия и web-версии пособия.
---
1. Что умеет скрипт
   Ищет аудиофайлы в указанной папке.
   Распознаёт речь через `faster-whisper`.
   Работает на CPU, включая ноутбуки без NVIDIA-видеокарты.
   Поддерживает быстрый CPU-профиль `int8`.
   Может предварительно готовить аудио через `ffmpeg`.
   Может передавать исходный файл напрямую в `faster-whisper` через `--skip-audio-prepare`.
   Сохраняет сырой транскрипт, таймкоды, SRT и JSON-сегменты.
   Очищает технический мусор, повторы и субтитровые артефакты.
   Выделяет важные сигналы ученика: «не понимаю», «почему», «не сходится» и похожие фразы.
   Создаёт смысловую версию транскрипта для будущего пособия.
   Делит текст на чанки для LLM.
   Генерирует готовый prompt-файл для ручной или автоматической LLM-обработки.
   Опционально запускает локальную LLM через Ollama.
   Сохраняет manifest и отчёты по обработке.
---
2. Рекомендуемая конфигурация для вашего ноутбука
   Для ноутбука с Intel Core i9-12900HK и 32 ГБ RAM:
```text
Whisper model: small
Device: cpu
Compute type: int8
Beam size: 1
VAD: включён
Content filter: medium
```
Быстрый рабочий запуск:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced `
  --content-filter-mode medium
```
---
3. Установка
   3.1. Создать виртуальное окружение
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Если PowerShell запрещает запуск скриптов:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
После этого заново активируйте окружение:
```powershell
.\.venv\Scripts\Activate.ps1
```
3.2. Установить зависимости
```powershell
pip install -U pip
pip install faster-whisper
```
Для Ollama-этапа дополнительные Python-зависимости обычно не нужны: скрипт обращается к Ollama через HTTP API стандартными средствами Python.
3.3. Установить ffmpeg
`ffmpeg` нужен для подготовки WAV, определения длительности и обработки аудио через фильтры.
На Windows удобно установить через `winget`:
```powershell
winget install Gyan.FFmpeg
```
Проверьте:
```powershell
ffmpeg -version
ffprobe -version
```
При запуске с `--skip-audio-prepare` скрипт передаёт исходный файл напрямую в `faster-whisper`. В этом режиме `ffmpeg` всё равно используется для определения длительности файла через `ffprobe`.
---
4. Минимальный requirements.txt
```text
faster-whisper
```
При желании можно зафиксировать версию:
```text
faster-whisper>=1.0.0
```
---
5. Быстрый запуск
   Папка должна содержать один или несколько аудиофайлов:
```text
C:\audio_lessons\
  lesson_01.mp3
  lesson_02.m4a
```
Команда:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced `
  --content-filter-mode medium
```
Рекурсивный поиск по вложенным папкам:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --recursive `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare
```
---
6. Режимы качества и скорости
   6.1. Самый быстрый CPU-профиль
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --content-filter-mode medium
```
6.2. Более аккуратный CPU-профиль
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 2 `
  --content-filter-mode medium
```
6.3. Качественнее, заметно медленнее
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model medium `
  --device cpu `
  --compute-type int8 `
  --beam-size 2 `
  --content-filter-mode medium
```
6.4. Мягкая нагрузка на ноутбук
Если ноутбук перегревается или хочется параллельно работать:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --cpu-threads 10 `
  --beam-size 1 `
  --skip-audio-prepare
```
Для максимальной скорости на вашем CPU можно оставить автоматическое значение: число логических ядер минус 1.
---
7. Смысловая фильтрация для будущего пособия
   Скрипт создаёт две важные версии текста:
```text
02_clean_balanced.txt
03_content_only_medium.txt
```
Назначение файлов:
Файл	Назначение
`02_clean_balanced.txt`	Полный очищенный транскрипт для проверки человеком
`03_content_only_medium.txt`	Смысловая версия для LLM и будущего пособия
Смысловая фильтрация удаляет организационные и малозначимые фразы:
```text
Здравствуйте.
Меня слышно?
Сейчас я всё подключу.
Видно экран?
Секундочку.
Окей.
Хорошо.
Ага.
Угу.
```
При этом скрипт старается сохранять педагогически значимые фрагменты:
```text
не понимаю
почему
у меня другой ответ
не сходится
можно ещё раз
задача
условие
решение
формула
логарифм
производная
уравнение
неравенство
```
7.1. Режимы `--content-filter-mode`
```text
off         — смысловая фильтрация выключена
safe        — удаляются самые очевидные служебные фразы
medium      — рабочий режим для подготовки пособий
aggressive  — сильное сокращение текста для очень длинных занятий
```
Рекомендуемый режим:
```powershell
--content-filter-mode medium
```
Бережный режим:
```powershell
--content-filter-mode safe
```
Жёсткий режим:
```powershell
--content-filter-mode aggressive
```
Для финальных учебных материалов желательно сверять спорные места с `02_clean_balanced.txt`.
---
8. Очистка транскрипта
   Параметр:
```powershell
--clean-mode conservative|balanced|aggressive
```
Режимы:
Режим	Назначение
`conservative`	Бережная очистка, сохраняет максимум текста

`balanced`	Рабочий режим по умолчанию
`aggressive`	Сильная очистка повторов и коротких вводных фраз
Рекомендуемая связка:
```powershell
--clean-mode balanced --content-filter-mode medium
```
---
9. Структура выходной папки
   Для файла:
```text
lesson_01.mp3
```
создаётся папка:
```text
lesson_01_lesson_transcript/
```
Пример содержимого:
```text
lesson_01_lesson_transcript/
  00_raw_whisper.txt
  00_raw_timestamped.txt
  00_raw_segments.srt
  00_raw_segments.json

  01_normalized.txt
  02_clean_balanced.txt
  03_content_only_medium.txt

  03_llm_chunks/
    chunk_001.txt
    chunk_002.txt
    chunk_003.txt

  04_llm_prompt.md
  important_student_signals.json
  cleaning_report.md
  manifest.json

  05_ollama_chunk_protocols/
    chunk_001_protocol.md
    chunk_001_ollama_response.json
    chunk_002_protocol.md
    chunk_002_ollama_response.json

  06_ollama_final_protocol.md
  06_ollama_final_prompt.md
  06_ollama_final_response.json
  ollama_report.json
```
Файлы Ollama появляются только при запуске с `--ollama`.
---
10. Какой файл использовать дальше
    Для будущего конвейера:
```text
аудио → транскрипт → LaTeX-пособие → web-пособие
```
лучше использовать такую схему:
```text
00_raw_whisper.txt
  → 02_clean_balanced.txt
  → 03_content_only_medium.txt
  → 03_llm_chunks/
  → учебный протокол
  → lesson_spec.json
  → LaTeX + web
```
Рекомендуемые файлы:
Задача	Файл
Проверка исходного смысла	`02_clean_balanced.txt`
Передача в LLM	`03_content_only_medium.txt` или `03_llm_chunks/`
Создание prompt вручную	`04_llm_prompt.md`
Поиск затруднений ученика	`important_student_signals.json`
Диагностика запуска	`manifest.json` и `cleaning_report.md`
---
11. Использование Ollama
    Ollama позволяет локально обработать смысловые чанки и получить учебный протокол занятия.
    11.1. Установка Ollama
    Установите Ollama с официального сайта, затем откройте новый PowerShell и проверьте:
```powershell
ollama --version
```
Проверка локального API:
```powershell
curl http://localhost:11434/api/tags
```
11.2. Рекомендуемые модели
Для CPU-ноутбука с 32 ГБ RAM:
```powershell
ollama pull qwen2.5:7b
```
Альтернатива:
```powershell
ollama pull qwen3:8b
```
Для стабильного рабочего пайплайна начните с `qwen2.5:7b`.
11.3. Запуск скрипта с Ollama
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced `
  --content-filter-mode medium `
  --ollama `
  --ollama-model qwen2.5:7b `
  --no-ollama-final-synthesis `
  --ollama-num-ctx 8192 `
  --ollama-num-predict 1200 `
  --chunk-size 7000 `
  --chunk-overlap 150
```
Для первой проверки лучше использовать `--no-ollama-final-synthesis`. Так скрипт создаст протоколы отдельных чанков и пропустит тяжёлую финальную сборку.
11.4. Запуск с итоговым протоколом
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced `
  --content-filter-mode medium `
  --ollama `
  --ollama-model qwen2.5:7b `
  --ollama-num-ctx 12000 `
  --ollama-num-predict 1800 `
  --chunk-size 8000 `
  --chunk-overlap 150
```
---
12. Основные параметры
    12.1. Faster-whisper
    Параметр	Значение по умолчанию	Назначение
    `--model`	`small`	Модель распознавания
    `--device`	`cpu`	Устройство: `cpu`, `cuda`, `auto`
    `--compute-type`	`int8`	Тип вычислений CTranslate2
    `--cpu-threads`	логические ядра минус 1	Количество CPU-потоков
    `--num-workers`	`1`	Worker'ы faster-whisper
    `--language`	`ru`	Язык речи
    `--beam-size`	`1`	Скорость/точность декодирования
    `--vad-filter`	включён	Пропуск участков без речи
    `--word-timestamps`	выключен	Пословные таймкоды
    12.2. Аудиоподготовка
    Параметр	Назначение
    `--skip-audio-prepare`	Передать исходный файл напрямую в faster-whisper
    `--audio-filter none`	Подготовить WAV без фильтров
    `--audio-filter basic`	Highpass/lowpass
    `--audio-filter loudnorm`	Нормализация громкости, медленнее
    `--keep-prepared-wav`	Сохранить подготовленный WAV
    12.3. Очистка и смысловая фильтрация
    Параметр	Назначение
    `--clean-mode`	Базовая очистка транскрипта
    `--content-filter-mode`	Смысловое сокращение перед LLM
    `--chunk-size`	Размер чанка для LLM
    `--chunk-overlap`	Перекрытие чанков
    `--prompt-include-text-limit`	Лимит вставки текста в `04_llm_prompt.md`
    12.4. Ollama
    Параметр	Значение по умолчанию	Назначение
    `--ollama`	выключен	Включить локальную LLM-обработку
    `--ollama-url`	`http://localhost:11434`	URL Ollama API
    `--ollama-model`	`qwen2.5:7b`	Модель Ollama
    `--ollama-timeout`	`900`	Таймаут одного запроса
    `--ollama-temperature`	`0.1`	Температура генерации
    `--ollama-num-ctx`	`8192`	Контекст модели
    `--ollama-num-predict`	`3000`	Максимальная длина ответа
    `--ollama-keep-alive`	`10m`	Время удержания модели в памяти
    `--no-ollama-final-synthesis`	выключен	Пропустить финальную сборку протокола
    `--ollama-final-max-chars`	`45000`	Лимит финального prompt
---
13. Типовые сценарии
    13.1. Только транскрипт и очищенный текст
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare
```
13.2. Транскрипт для будущего пособия
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced `
  --content-filter-mode medium `
  --chunk-size 7000 `
  --chunk-overlap 150
```
13.3. Полный локальный протокол через Ollama
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced `
  --content-filter-mode medium `
  --ollama `
  --ollama-model qwen2.5:7b `
  --ollama-num-ctx 8192 `
  --ollama-num-predict 1500 `
  --chunk-size 7000 `
  --chunk-overlap 150
```
---
14. Диагностика
    14.1. Модель faster-whisper скачивается с Hugging Face
    При первом запуске возможны строки вида:
```text
HTTP Request: GET https://huggingface.co/...
Warning: You are sending unauthenticated requests to the HF Hub.
```
Это нормальная ситуация первого скачивания модели. После загрузки модель попадёт в кэш Hugging Face.
14.2. Предупреждение про symlinks на Windows
Возможен warning:
```text
cache-system uses symlinks by default ... your machine does not support them
```
Варианты:
Включить Developer Mode в Windows.
Запускать терминал от администратора.
Отключить предупреждение:
```powershell
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
```
После `setx` откройте новый PowerShell.
14.3. ffmpeg найден, ffprobe отсутствует
Проверьте:
```powershell
where ffmpeg
where ffprobe
```
Если команды ничего не выводят, добавьте папку `bin` от ffmpeg в `PATH`.
14.4. Ollama недоступна
Проверьте:
```powershell
ollama list
curl http://localhost:11434/api/tags
```
Если модель отсутствует:
```powershell
ollama pull qwen2.5:7b
```
14.5. Ноутбук сильно загружен
Снизьте число CPU-потоков:
```powershell
--cpu-threads 8
```
или:
```powershell
--cpu-threads 10
```
14.6. Транскрипт слишком сильно сокращён
Используйте более мягкий режим:
```powershell
--content-filter-mode safe
```
Или отключите смысловую фильтрацию:
```powershell
--content-filter-mode off
```
14.7. В протоколе потерялись важные реплики ученика
Проверьте файл:
```text
important_student_signals.json
```
Затем сверяйтесь с:
```text
02_clean_balanced.txt
```
---
15. Рекомендованный .gitignore
```gitignore
.venv/
__pycache__/
*.pyc

*_lesson_transcript/
input/audio/
output/

*.mp3
*.wav
*.m4a
*.flac
*.ogg
*.aac
*.wma
```
Если аудиофайлы должны храниться в репозитории, уберите аудио-расширения из `.gitignore`.
---
16. Рекомендуемый дальнейший конвейер
    После текущего скрипта следующий инженерный шаг — создать единый учебный JSON:
```text
03_content_only_medium.txt
  → lesson_protocol.md
  → lesson_spec.json
  → main.tex
  → main.pdf
  → index.html
```
Идеальная схема:
```text
аудио
  → faster-whisper
  → cleaned transcript
  → content-only transcript
  → lesson protocol
  → lesson_spec.json
  → LaTeX-пособие
  → web-пособие
```
`lesson_spec.json` должен стать единым источником содержания для PDF и web-версии.
---
17. Короткая памятка
    Для обычной работы используйте:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced `
  --content-filter-mode medium
```
Для локального протокола через Ollama:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced `
  --content-filter-mode medium `
  --ollama `
  --ollama-model qwen2.5:7b `
  --no-ollama-final-synthesis `
  --ollama-num-predict 1200 `
  --chunk-size 7000 `
  --chunk-overlap 150
```