# Здоровье под контролем ИИ — выявление и локализация опухолей на МРТ головного мозга

Трек «Студенты», Медицина 2050. Классы: глиома, менингиома, опухоль гипофиза, нет опухоли.
Система — вспомогательный инструмент для врача, диагноз не ставит.

## Требования
- **Python 3.10, 3.11 или 3.12** (на 3.13/3.14 ML-библиотеки могут не установиться).
- Для обучения нужна видеокарта: используем Google Colab / Kaggle. Локально хватает аудита данных и прототипа.

## Быстрый старт

### Вариант А. PyCharm (Windows)
1. **File → Open** → папка проекта (где лежит этот README).
2. **File → Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Virtualenv Environment → New**.
   Base interpreter — Python 3.12. Location оставить `.venv` внутри проекта.
3. Открыть терминал PyCharm (Alt+F12). В начале строки должно быть `(.venv)`. Выполнить:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements-base.txt
   ```
   Этого достаточно для аудита данных и тестов. Полный набор (torch и др., несколько ГБ) ставим, когда доходим до обучения:
   ```
   pip install -r requirements.txt
   ```
4. Правый клик по папке `src` → **Mark Directory as → Sources Root**.
   Правый клик по `data/raw` → **Mark Directory as → Excluded**.

### Вариант Б. Терминал
Windows (PowerShell):
```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-base.txt
```
Если PowerShell запрещает скрипты: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Linux / macOS:
```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-base.txt
```

## Данные
Данные в git не храним. Скачать архив по ссылке из презентации трека и распаковать так:
```
data/raw/train/<класс>/*.jpg
data/raw/test/<класс>/*.jpg
```

## Первые команды
Запускать из корня проекта:
```
python scripts/audit_data.py --root data/raw --out data/manifest.csv
pytest
```
Скрипт аудита печатает: число файлов, битые файлы, типы аугментаций в именах, число уникальных исходных снимков,
дубликаты и пересечение train/test (утечка данных). Результат нужно показать команде.


## Прототип (Gradio)
Веса и пороги лежат в `weights/<запуск>/` (создаёт блокнот обучения и блокнот 09).
```
pip install -r requirements.txt
python app/gradio_app.py --weights weights/convnext_tiny_base --examples examples
```
Интерфейс показывает класс, оценку модели по классам, «светофор доверия», карту внимания и область интереса.
Это исследовательский прототип, а не медицинское изделие. Публичную ссылку (`--share`) использовать только с обезличенными данными.

## Структура
- `src/brainmri/data` — манифест, сплиты без утечки (по исходному снимку)
- `src/brainmri/models` — классификатор (timm), сегментация (U-Net / MedSAM)
- `src/brainmri/explain` — Grad-CAM, псевдомаски
- `app/` — прототип (Gradio)
- `docs/` — безопасность, нормативка, схема внедрения
- `reports/` — метрики, графики, разбор ошибок
- `configs/` — конфиги экспериментов

Правила работы с git: см. `CONTRIBUTING.md`.

## Частые проблемы
- **`Unable to copy ... venvlauncher.exe`** — закрыть PyCharm, вручную удалить папку `.venv`, создать заново (лучше на Python 3.12).
- **`ModuleNotFoundError: brainmri`** — пометить `src` как Sources Root (шаг 4) или запускать скрипты из корня проекта.
- **`pip install` падает на torch/opencv** — проверить версию Python (`python --version` должна быть 3.10–3.12).
