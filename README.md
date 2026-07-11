# lesson_transcriber_faster.py

Скрипт для локальной транскрибации учебных аудиозаписей через `faster-whisper`, очистки транскрипта, смысловой фильтрации и опциональной методической обработки через Ollama.

Основной сценарий:

```text
аудио занятия → сырой транскрипт → очищенный транскрипт → смысловая версия → чанки для LLM → протокол занятия
```

Скрипт рассчитан на Windows + PowerShell и хорошо подходит для CPU-ноутбука без NVIDIA-видеокарты.

---

## 1. Быстрый старт

Перейдите в папку проекта:

```powershell
cd "C:\Users\Артем\IdeaProjects\audio-transcription"
```

Проверьте синтаксис скрипта:

```powershell
python -m py_compile .\lesson_transcriber_faster.py
```

Проверьте, что в справке доступны короткие ключи:

```powershell
python .\lesson_transcriber_faster.py -h | findstr /C:"--fast" /C:"--llm" /C:"--llm-final"
```

---

## 2. Короткая команда без LLM

Для обычной транскрибации, очистки и смысловой фильтрации используйте:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --fast
```

Ключ `--fast` применяет рабочий CPU-профиль:

```text
--model small
--device cpu
--compute-type int8
--beam-size 1
--skip-audio-prepare
--clean-mode balanced
--content-filter-mode medium
```

Этот режим создаёт полный транскрипт, очищенный транскрипт и смысловую версию для дальнейшей генерации пособия.

---

## 3. Короткая команда с LLM/Ollama

Для транскрибации с последующей обработкой через локальную LLM:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --llm
```

Ключ `--llm` включает профиль `--fast` и добавляет Ollama-настройки:

```text
--ollama
--ollama-model qwen2.5:7b
--no-ollama-final-synthesis
--ollama-num-predict 1200
--chunk-size 7000
--chunk-overlap 150
```

Такой режим создаёт протоколы отдельных чанков в папке `05_ollama_chunk_protocols`.

---

## 4. Короткая команда с LLM и итоговым протоколом

Для транскрибации, обработки чанков и сборки единого итогового протокола используйте:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --llm-final
```

Ключ `--llm-final` включает профиль `--fast`, запускает Ollama и оставляет финальную сборку включённой. На выходе ожидается файл:

```text
06_ollama_final_protocol.md
```

Также сохраняются:

```text
06_ollama_final_prompt.md
06_ollama_final_response.json
```

Если итоговый prompt окажется слишком большим, в `06_ollama_final_protocol.md` будет записано предупреждение. В таком случае используйте протоколы отдельных чанков из `05_ollama_chunk_protocols/` или увеличьте `--ollama-final-max-chars` и `--ollama-num-ctx`.

---

## 5. Полная команда без LLM

Эквивалент `--fast` в развёрнутом виде:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --model small --device cpu --compute-type int8 --beam-size 1 --skip-audio-prepare --clean-mode balanced --content-filter-mode medium
```

---

## 6. Полная команда с LLM

Эквивалент `--llm` в развёрнутом виде:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --model small --device cpu --compute-type int8 --beam-size 1 --skip-audio-prepare --clean-mode balanced --content-filter-mode medium --ollama --ollama-model qwen2.5:7b --no-ollama-final-synthesis --ollama-num-predict 1200 --chunk-size 7000 --chunk-overlap 150
```

---

## 7. Установка зависимостей

Минимальные зависимости:

```powershell
pip install faster-whisper
```

Для работы с Ollama установите Ollama отдельно и скачайте модель:

```powershell
ollama pull qwen2.5:7b
```

Проверка Ollama:

```powershell
ollama list
curl http://localhost:11434/api/tags
```

---

## 8. Рекомендуемые параметры для CPU-ноутбука

Для ноутбука без NVIDIA-видеокарты:

```text
model: small
mode: cpu
compute_type: int8
beam_size: 1
vad_filter: включён
skip_audio_prepare: включён
content_filter_mode: medium
```

Для более качественного распознавания можно попробовать:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --model medium --device cpu --compute-type int8 --beam-size 2 --skip-audio-prepare --clean-mode balanced --content-filter-mode medium
```

Модель `medium` потребует больше времени.

---

## 9. Что делает смысловая фильтрация

Ключ:

```powershell
--content-filter-mode medium
```

создаёт компактную смысловую версию транскрипта для дальнейшей LLM-обработки.

Режимы:

| Режим | Назначение |
|---|---|
| `off` | смысловая фильтрация выключена |
| `safe` | удаляются очевидные технические и организационные фразы |
| `medium` | рабочий режим для будущего пособия |
| `aggressive` | сильное сокращение текста для длинных занятий |

Смысловая фильтрация сохраняет педагогически важные фрагменты: задачи, формулы, объяснения, ошибки, затруднения ученика, просьбы повторить, места с неуверенностью.

---

## 10. Структура выходной папки

Для файла `lesson.mp3` будет создана папка:

```text
lesson_lesson_transcript/
```

Внутри появятся файлы:

```text
00_raw_whisper.txt              сырой текст Whisper
00_raw_timestamped.txt          сырой текст с таймкодами
00_raw_segments.srt             субтитры SRT
00_raw_segments.json            сегменты распознавания
01_normalized.txt               нормализованный текст
02_clean_balanced.txt           полный очищенный транскрипт
03_content_only_medium.txt      смысловая версия для пособия
03_llm_chunks/                  чанки для LLM
04_llm_prompt.md                prompt для ручной LLM-обработки
important_student_signals.json  найденные сигналы затруднений ученика
cleaning_report.md              отчёт об очистке
manifest.json                   параметры запуска и тайминги
```

При запуске с `--llm` дополнительно:

```text
05_ollama_chunk_protocols/      протоколы отдельных чанков
ollama_report.json              отчёт Ollama-этапа
```

При включённой финальной сборке появятся:

```text
06_ollama_final_protocol.md
06_ollama_final_prompt.md
06_ollama_final_response.json
```

В коротком профиле `--llm` финальная сборка отключена через `--no-ollama-final-synthesis`, чтобы ускорить локальную обработку.

---

## 11. Какой файл использовать дальше

Для генерации учебного пособия используйте:

```text
03_content_only_medium.txt
```

Для проверки спорных мест используйте:

```text
02_clean_balanced.txt
```

Для точной сверки по времени используйте:

```text
00_raw_timestamped.txt
00_raw_segments.srt
```

---

## 12. Как работает `--skip-audio-prepare`

Ключ:

```powershell
--skip-audio-prepare
```

передаёт исходный аудиофайл напрямую в `faster-whisper`. Это обычно быстрее, поскольку скрипт пропускает создание промежуточного WAV через `ffmpeg`.

Если аудио плохого качества, можно выполнить обработку через `ffmpeg`:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --model small --device cpu --compute-type int8 --beam-size 1 --clean-mode balanced --content-filter-mode medium --audio-filter basic
```

Доступные режимы аудиофильтра:

```text
none      без фильтров
basic     highpass/lowpass
loudnorm  нормализация громкости, медленнее
```

---

## 13. Частые проблемы

### PowerShell вставляет многострочную команду кусками

Используйте короткий профиль:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --fast
```

или одну строку без обратных кавычек.

### `unrecognized arguments: --content-filter-mode medium`

Запущена старая версия скрипта. Проверьте:

```powershell
python .\lesson_transcriber_faster.py -h | findstr content-filter
```

В справке должна быть строка:

```text
--content-filter-mode {off,safe,medium,aggressive}
```

### `unrecognized arguments: --fast`, `--llm` или `--llm-final`

Запущена версия без коротких профилей. Проверьте:

```powershell
python .\lesson_transcriber_faster.py -h | findstr /C:"--fast" /C:"--llm" /C:"--llm-final"
```

### `ModuleNotFoundError: No module named 'faster_whisper'`

Установите зависимость:

```powershell
pip install faster-whisper
```

### Ollama не найдена

Проверьте:

```powershell
ollama --version
ollama list
```

### Модель Ollama отсутствует

Скачайте модель:

```powershell
ollama pull qwen2.5:7b
```

---

## 14. Рекомендуемый рабочий процесс

1. Поместите аудио в папку:

```text
C:\Users\Артем\IdeaProjects\audio-transcription\audio
```

2. Запустите быструю транскрибацию:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --fast
```

3. Проверьте файл:

```text
03_content_only_medium.txt
```

4. Запустите LLM-обработку по чанкам:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --llm
```

5. Для единого итогового протокола запустите:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --llm-final
```

6. Используйте `06_ollama_final_protocol.md` или файлы из:

```text
05_ollama_chunk_protocols/
```

как основу для генерации `lesson_spec.json`, LaTeX-пособия и web-версии.

---

## 15. Git-команды после обновления

```powershell
git status
git add lesson_transcriber_faster.py README.md
git commit -m "Add short launch profiles and update README"
git push
```

---

## 16. Краткая памятка

Только транскрипт и смысловая фильтрация:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --fast
```

Транскрипт плюс Ollama по чанкам:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --llm
```

Транскрипт плюс Ollama с единым итоговым протоколом:

```powershell
python .\lesson_transcriber_faster.py "C:\Users\Артем\IdeaProjects\audio-transcription\audio" --llm-final
```
