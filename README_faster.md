Audio Lesson Transcriber Faster
Скрипт для транскрибации учебных аудиозаписей через `faster-whisper` с последующей очисткой текста, разбиением на чанки для LLM и опциональной методической обработкой через локальную Ollama-модель.
Основной сценарий: запись занятия по математике, физике или химии → сырой транскрипт → очищенный транскрипт → prompt/чанки для создания учебного протокола, пособия или анализа ошибок ученика.
---
1. Что умеет скрипт
   Обрабатывает аудиофайлы из указанной папки.
   Поддерживает форматы: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.wma`.
   Работает через `faster-whisper`, а не через `openai-whisper`.
   Оптимизирован для CPU-режима:
   `device=cpu`;
   `compute_type=int8`;
   настраиваемое число CPU-потоков.
   Может читать исходный аудиофайл напрямую без предварительного WAV через `ffmpeg`.
   Может предварительно готовить аудио через `ffmpeg`:
   без фильтров;
   с базовым `highpass/lowpass`;
   с `loudnorm`.
   Использует VAD-фильтр для пропуска тишины.
   Сохраняет:
   сырой текст;
   текст с таймкодами;
   SRT-субтитры;
   JSON сегментов;
   очищенный текст;
   чанки для LLM;
   prompt-файл для дальнейшей обработки;
   отчёт об очистке;
   manifest с настройками и временем выполнения.
   Опционально запускает Ollama для методической обработки чанков.
---
2. Требования
   Python
   Рекомендуется Python 3.10+.
   Проверка версии:
```bash
python --version
```
Основные Python-зависимости
```bash
pip install faster-whisper
```
Дополнительный `openai-whisper` для этого скрипта не нужен.
FFmpeg
`ffmpeg` нужен только если вы не используете `--skip-audio-prepare`.
При запуске с `--skip-audio-prepare` скрипт отдаёт исходный аудиофайл напрямую в `faster-whisper`. В этом режиме `ffmpeg` обычно не требуется для самой транскрибации, но `ffprobe` полезен для определения длительности файла до обработки.
Проверка:
```bash
ffmpeg -version
ffprobe -version
```
---
3. Быстрый старт
   Положите аудиофайлы в отдельную папку, например:
```text
C:\audio_lessons
```
Запуск быстрого CPU-профиля:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare
```
Это рекомендуемый стартовый вариант для ноутбука без NVIDIA-видеокарты.
---
4. Рекомендуемые профили запуска
   4.1. Максимально быстрый CPU-профиль
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare
```
Подходит для первичной транскрибации занятий, когда скорость важнее идеальной точности.
4.2. Более точный CPU-профиль
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model medium `
  --device cpu `
  --compute-type int8 `
  --beam-size 2 `
  --skip-audio-prepare
```
Подходит для сложных записей, где много формул, терминов, неразборчивой речи или шума. Будет заметно медленнее, чем `small`.
4.3. Профиль с предварительной обработкой аудио через FFmpeg
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --audio-filter basic
```
В этом режиме создаётся временный WAV 16 kHz mono. Фильтр `basic` применяет `highpass=f=80,lowpass=f=7600`.
4.4. Профиль с нормализацией громкости
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --audio-filter loudnorm
```
Этот режим может помочь при очень неровной громкости, но он медленнее.
4.5. Обработка вложенных папок
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
5. Запуск с Ollama
   Ollama-этап нужен, если после транскрибации вы хотите автоматически получить методический протокол занятия.
   Перед запуском убедитесь, что Ollama работает и нужная модель скачана:
```bash
ollama list
ollama pull qwen2.5:7b
```
Быстрый вариант без финальной сборки общего протокола:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --ollama `
  --ollama-model qwen2.5:7b `
  --no-ollama-final-synthesis `
  --ollama-num-predict 1200 `
  --chunk-size 8000 `
  --chunk-overlap 150
```
Более полный вариант с финальной сборкой общего протокола:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --ollama `
  --ollama-model qwen2.5:7b
```
Финальная сборка может быть долгой и может не выполниться, если итоговый prompt окажется больше лимита `--ollama-final-max-chars`.
---
6. Структура результата
   Для каждого файла создаётся папка рядом с исходным аудио:
```text
<имя_аудиофайла>_lesson_transcript/
```
Пример:
```text
lesson_01.mp3
lesson_01_lesson_transcript/
```
Внутри папки результата:
```text
00_raw_whisper.txt
00_raw_timestamped.txt
00_raw_segments.srt
00_raw_segments.json
01_normalized.txt
02_clean_balanced.txt
03_llm_chunks/
04_llm_prompt.md
important_student_signals.json
cleaning_report.md
manifest.json
```
Если включена Ollama:
```text
05_ollama_chunk_protocols/
06_ollama_final_protocol.md
06_ollama_final_prompt.md
06_ollama_final_response.json
ollama_report.json
```
Если итоговая сборка отключена через `--no-ollama-final-synthesis`, файла `06_ollama_final_protocol.md` может не быть.
---
7. Основные выходные файлы
   `00_raw_whisper.txt`
   Сырой текст, полученный от `faster-whisper`.
   `00_raw_timestamped.txt`
   Сегменты с таймкодами в человекочитаемом виде.
   `00_raw_segments.srt`
   SRT-файл субтитров.
   `00_raw_segments.json`
   JSON со списком сегментов:
   начало;
   конец;
   текст;
   `avg_logprob`;
   `no_speech_prob`;
   `compression_ratio`.
   `01_normalized.txt`
   Нормализованный текст: исправлены пробелы, переносы, базовая пунктуационная структура.
   `02_clean_<mode>.txt`
   Очищенный текст после выбранного режима очистки:
   `conservative`;
   `balanced`;
   `aggressive`.
   `03_llm_chunks/`
   Папка с чанками очищенного транскрипта для дальнейшей LLM-обработки.
   `04_llm_prompt.md`
   Готовый prompt для ручной или автоматической обработки транскрипта через LLM.
   `cleaning_report.md`
   Отчёт об очистке:
   размер сырого текста;
   размер очищенного текста;
   количество чанков;
   найденные сигналы затруднений ученика;
   рекомендации, что делать дальше.
   `manifest.json`
   Технический манифест обработки:
   исходный файл;
   длительность;
   настройки `faster-whisper`;
   настройки предобработки;
   настройки Ollama;
   время выполнения этапов;
   пути к результатам.
---
8. Важные параметры faster-whisper
   `--model`
   Модель распознавания.
   Примеры:
```text
tiny
base
small
medium
large-v3
large-v3-turbo
turbo
```
Рекомендация для CPU-ноутбука:
`small` — основной рабочий вариант;
`medium` — если нужно качество выше и можно ждать;
`base` — если нужна максимальная скорость;
`large-v3` — обычно слишком тяжело для CPU-сценария.
`--device`
```text
cpu
cuda
auto
```
Для ноутбука без NVIDIA:
```powershell
--device cpu
```
`--compute-type`
Для CPU рекомендуется:
```powershell
--compute-type int8
```
Для CUDA обычно используют:
```powershell
--compute-type float16
```
или:
```powershell
--compute-type int8_float16
```
`--cpu-threads`
Количество CPU-потоков для `faster-whisper`.
По умолчанию скрипт берёт число логических ядер минус 1.
Пример ручной настройки:
```powershell
--cpu-threads 8
```
Если ноутбук начинает сильно греться или тормозить, уменьшите значение:
```powershell
--cpu-threads 4
```
`--num-workers`
Количество worker'ов `faster-whisper`.
Для последовательной обработки файлов обычно оставляйте:
```powershell
--num-workers 1
```
`--beam-size`
Ширина beam search.
`1` — быстрее;
`2` — разумный компромисс;
`5` — потенциально точнее, но медленнее.
Рекомендация:
```powershell
--beam-size 1
```
или:
```powershell
--beam-size 2
```
---
9. VAD-фильтр
   По умолчанию VAD включён:
```powershell
--vad-filter
```
Он помогает пропускать тишину и длинные участки без речи.
Отключение:
```powershell
--no-vad-filter
```
Настройки:
```powershell
--vad-min-silence-duration-ms 1000
--vad-speech-pad-ms 300
```
Если VAD «съедает» начало или конец реплик, увеличьте отступ:
```powershell
--vad-speech-pad-ms 500
```
Если в аудио много пауз и хочется агрессивнее вырезать тишину:
```powershell
--vad-min-silence-duration-ms 500
```
---
10. Предварительная подготовка аудио
    Быстрый режим без подготовки
```powershell
--skip-audio-prepare
```
Скрипт отдаёт исходный файл напрямую в `faster-whisper`.
Плюсы:
быстрее;
не нужен временный WAV;
проще пайплайн.
Минусы:
не применяются аудиофильтры;
при проблемных файлах может быть полезнее подготовка через `ffmpeg`.
Подготовка через FFmpeg
Не указывайте `--skip-audio-prepare`.
Тогда скрипт создаст временный WAV:
```text
16 kHz, mono, pcm_s16le
```
Фильтры:
```powershell
--audio-filter none
--audio-filter basic
--audio-filter loudnorm
```
Рекомендации:
`none` — быстрее всего среди ffmpeg-режимов;
`basic` — рабочий компромисс;
`loudnorm` — для неровной громкости, но медленнее.
---
11. Режимы очистки текста
    `conservative`
    Бережная очистка.
    Используйте, если важно ничего не потерять:
```powershell
--clean-mode conservative
```
`balanced`
Рабочий режим по умолчанию:
```powershell
--clean-mode balanced
```
`aggressive`
Сильная очистка:
```powershell
--clean-mode aggressive
```
Подходит, если в транскрипте много мусора, но есть риск удалить короткие педагогически значимые реплики вроде «да», «нет», «хорошо».
---
12. Чанки для LLM
    Параметры:
```powershell
--chunk-size 6000
--chunk-overlap 500
```
Для ускорения Ollama-этапа можно увеличить размер чанка и уменьшить перекрытие:
```powershell
--chunk-size 8000 `
--chunk-overlap 150
```
Для более аккуратной обработки:
```powershell
--chunk-size 5000 `
--chunk-overlap 500
```
---
13. Initial prompt
    По умолчанию используется подсказка для русскоязычных учебных занятий с терминами по математике, физике и химии.
    Отключить её можно так:
```powershell
--initial-prompt ""
```
Задать свою:
```powershell
--initial-prompt "Это занятие по стереометрии ЕГЭ. Часто встречаются слова: пирамида, призма, сечение, перпендикуляр, угол, расстояние."
```
---
14. `condition_on_previous_text`
    По умолчанию выключено.
    Включение:
```powershell
--condition-on-previous-text
```
Иногда это улучшает связность текста, но может усиливать повторы и ложные продолжения. Для длинных учебных записей обычно лучше начинать без этого флага.
---
15. Подробный лог
```powershell
--verbose
```
В подробном режиме будет больше информации о командах, этапах обработки и работе `faster-whisper`.
---
16. Примеры полных команд
    Один рабочий быстрый запуск
```powershell
python audio_lesson_transcriber_faster.py "C:\Users\Artem\Lessons\audio" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --clean-mode balanced
```
Запуск с сохранением подготовленного WAV
```powershell
python audio_lesson_transcriber_faster.py "C:\Users\Artem\Lessons\audio" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --audio-filter basic `
  --keep-prepared-wav
```
Флаг `--keep-prepared-wav` работает только если не указан `--skip-audio-prepare`.
Запуск для сложной записи
```powershell
python audio_lesson_transcriber_faster.py "C:\Users\Artem\Lessons\audio" `
  --model medium `
  --device cpu `
  --compute-type int8 `
  --beam-size 2 `
  --audio-filter basic `
  --clean-mode conservative
```
Запуск с Ollama и быстрым протоколированием чанков
```powershell
python audio_lesson_transcriber_faster.py "C:\Users\Artem\Lessons\audio" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare `
  --ollama `
  --ollama-model qwen2.5:7b `
  --no-ollama-final-synthesis `
  --ollama-num-predict 1200 `
  --chunk-size 8000 `
  --chunk-overlap 150
```
---
17. Как читать скорость обработки
    В логах скрипт выводит примерно такие строки:
```text
faster-whisper завершил транскрибацию за 320.45 сек; скорость: 1.87x realtime
```
`1.87x realtime` означает, что аудио обработано примерно в 1.87 раза быстрее его длительности.
Пример:
запись длится 60 минут;
скорость `2.00x realtime`;
транскрибация заняла примерно 30 минут.
В `manifest.json` скорость сохраняется в поле:
```json
"timings_seconds": {
  "realtime_factor": 1.87
}
```
---
18. Типовые проблемы и решения
    Ошибка: `ModuleNotFoundError: No module named 'faster_whisper'`
    Установите пакет:
```bash
pip install faster-whisper
```
Проверьте, что установка выполнена в том же виртуальном окружении, из которого запускается скрипт.
Ошибка: `ffmpeg не найден в PATH`
У вас два варианта.
Первый — запускать без подготовки аудио:
```powershell
--skip-audio-prepare
```
Второй — установить FFmpeg и добавить его в PATH.
Скрипт работает слишком медленно
Попробуйте:
```powershell
--model small `
--beam-size 1 `
--compute-type int8 `
--skip-audio-prepare
```
Также можно уменьшить число потоков, если ноутбук перегревается:
```powershell
--cpu-threads 4
```
Или увеличить, если процессор недогружен:
```powershell
--cpu-threads 8
```
Слишком много мусора в транскрипте
Попробуйте:
```powershell
--clean-mode aggressive
```
Или включите подготовку аудио:
```powershell
--audio-filter basic
```
Если громкость сильно скачет:
```powershell
--audio-filter loudnorm
```
Очистка удалила слишком много
Используйте более бережный режим:
```powershell
--clean-mode conservative
```
Ollama работает слишком долго
Отключите финальную сборку:
```powershell
--no-ollama-final-synthesis
```
Сократите ответ модели:
```powershell
--ollama-num-predict 1200
```
Уменьшите перекрытие чанков:
```powershell
--chunk-overlap 150
```
Модель Ollama не найдена
Проверьте список моделей:
```bash
ollama list
```
Скачайте нужную:
```bash
ollama pull qwen2.5:7b
```
Или укажите установленную модель:
```powershell
--ollama-model llama3.1:8b
```
---
19. Практический порядок работы
    Сначала запустите быстрый режим:
```powershell
python audio_lesson_transcriber_faster.py "C:\audio_lessons" `
  --model small `
  --device cpu `
  --compute-type int8 `
  --beam-size 1 `
  --skip-audio-prepare
```
Откройте файл:
```text
02_clean_balanced.txt
```
Если качество нормальное, используйте:
```text
04_llm_prompt.md
```
или чанки из:
```text
03_llm_chunks/
```
Если качество плохое, повторите с одним из вариантов:
```powershell
--model medium
```
или:
```powershell
--audio-filter basic
```
или:
```powershell
--clean-mode conservative
```
Если нужен автоматический протокол, добавьте Ollama:
```powershell
--ollama --no-ollama-final-synthesis
```
---
20. Рекомендуемые настройки для ноутбука без NVIDIA и с 32 ГБ RAM
    Базовый вариант:
```powershell
--model small
--device cpu
--compute-type int8
--beam-size 1
--skip-audio-prepare
--clean-mode balanced
```
Качественный вариант:
```powershell
--model medium
--device cpu
--compute-type int8
--beam-size 2
--skip-audio-prepare
--clean-mode conservative
```
Вариант для проблемного звука:
```powershell
--model small
--device cpu
--compute-type int8
--beam-size 1
--audio-filter basic
--clean-mode balanced
```
---
21. Краткая справка по аргументам
```text
input_dir                         Папка с аудиофайлами
--model                          Модель faster-whisper
--device                         cpu / cuda / auto
--compute-type                   int8 / float16 / int8_float16 и др.
--cpu-threads                    Число CPU-потоков
--num-workers                    Число worker'ов faster-whisper
--language                       Язык речи, по умолчанию ru
--beam-size                      Beam search, по умолчанию 1
--vad-filter / --no-vad-filter   Включить или отключить VAD
--vad-min-silence-duration-ms    Минимальная тишина для VAD
--vad-speech-pad-ms              Отступ вокруг речи для VAD
--word-timestamps                Пословные таймкоды
--skip-audio-prepare             Не готовить WAV через ffmpeg
--audio-filter                   none / basic / loudnorm
--clean-mode                     conservative / balanced / aggressive
--chunk-size                     Размер чанка для LLM
--chunk-overlap                  Перекрытие чанков
--prompt-include-text-limit      Лимит вставки текста в 04_llm_prompt.md
--initial-prompt                 Начальная подсказка для Whisper
--condition-on-previous-text     Связывать распознавание с предыдущим текстом
--keep-prepared-wav              Сохранять подготовленный WAV
--recursive                      Искать файлы во вложенных папках
--verbose                        Подробный лог
--ollama                         Включить Ollama-обработку
--ollama-url                     URL Ollama API
--ollama-model                   Модель Ollama
--ollama-timeout                 Таймаут запроса к Ollama
--ollama-temperature             Температура Ollama
--ollama-num-ctx                 Контекст Ollama
--ollama-num-predict             Максимум генерируемых токенов
--ollama-keep-alive              Время удержания модели в памяти
--no-ollama-final-synthesis      Не собирать итоговый протокол
--ollama-final-max-chars         Лимит размера финального prompt
```
---
22. Минимальный `.gitignore`
    Рекомендуется не коммитить результаты обработки и большие аудиофайлы:
```gitignore
*.mp3
*.wav
*.m4a
*.flac
*.ogg
*.aac
*.wma
*_lesson_transcript/
__pycache__/
.venv/
venv/
```
---
23. Минимальный `requirements.txt`
```text
faster-whisper
```
Если проект использует дополнительные инструменты, добавьте их отдельно.
---
24. Назначение файлов в рабочем процессе преподавателя
    После обработки занятия обычно используются три файла:
    `02_clean_balanced.txt` — быстрая проверка очищенного транскрипта.
    `04_llm_prompt.md` — готовая заготовка для создания учебного протокола.
    `03_llm_chunks/` — фрагменты для последовательной обработки большой записи.
    Для последующего создания персонального пособия лучше опираться не только на итоговый LLM-протокол, но и на сырой файл с таймкодами `00_raw_timestamped.txt`, особенно если в занятии были важные ошибки ученика или неразборчивые формулы.