# Инструкция по обучению модели на Cloud.ru

## Подготовка

### 1. Установка зависимостей для обучения

```bash
pip install -r requirements-train.txt
```

### 2. Подготовка данных

Убедитесь, что файл с вакансиями (`IT_vacancies_full 2.csv`) доступен на сервере Cloud.ru.

### 3. Настройка переменных окружения

Добавьте в `.env`:
```env
MODEL_NAME=IlyaGusev/saiga_mistral_7b_merged
VACANCIES_CSV_PATH=IT_vacancies_full 2.csv
```

## Обучение на Cloud.ru

### Вариант 1: Обучение через Jupyter Notebook

1. Создайте Jupyter Notebook на Cloud.ru
2. Загрузите файлы проекта
3. Запустите ячейку:

```python
!python train_model.py \
    --model-name IlyaGusev/saiga_mistral_7b_merged \
    --csv-path "IT_vacancies_full 2.csv" \
    --output-dir ./models/finetuned \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --max-samples 1000
```

### Вариант 2: Обучение через SSH/терминал

1. Подключитесь к инстансу Cloud.ru с GPU
2. Перейдите в директорию проекта
3. Запустите обучение:

```bash
python train_model.py \
    --model-name IlyaGusev/saiga_mistral_7b_merged \
    --csv-path "IT_vacancies_full 2.csv" \
    --output-dir ./models/finetuned \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5
```

### Вариант 3: Обучение через Cloud.ru ML Platform

1. Создайте задачу обучения в ML Platform
2. Укажите:
   - **Скрипт**: `train_model.py`
   - **Зависимости**: `requirements-train.txt`
   - **Данные**: путь к CSV файлу
   - **GPU**: выберите GPU инстанс
   - **Аргументы**: см. параметры ниже

## Параметры обучения

### Основные параметры

- `--model-name`: Базовая модель (по умолчанию: `IlyaGusev/saiga_mistral_7b_merged`)
- `--csv-path`: Путь к CSV файлу с вакансиями
- `--output-dir`: Директория для сохранения модели (по умолчанию: `./models/finetuned`)
- `--num-epochs`: Количество эпох (по умолчанию: 3)
- `--batch-size`: Размер батча (по умолчанию: 4, зависит от GPU памяти)
- `--learning-rate`: Learning rate (по умолчанию: 2e-5)
- `--max-samples`: Максимальное количество примеров (опционально, для быстрого теста)
- `--max-length`: Максимальная длина последовательности (по умолчанию: 512)

### Рекомендации по параметрам

**Для GPU с 16GB памяти:**
```bash
--batch-size 4 --gradient-accumulation-steps 4
```

**Для GPU с 24GB+ памяти:**
```bash
--batch-size 8 --gradient-accumulation-steps 2
```

**Для быстрого теста:**
```bash
--max-samples 100 --num-epochs 1
```

## Мониторинг обучения

Во время обучения вы увидите:
- Прогресс по эпохам
- Loss на train и validation
- Время обучения
- Использование GPU

## Использование обученной модели

После обучения модель будет сохранена в указанной директории. Для использования:

1. Обновите `.env`:
```env
EVOLUTION_MODEL=./models/finetuned
```

2. Или используйте напрямую в коде:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/finetuned")
model = AutoModelForCausalLM.from_pretrained("./models/finetuned")
```

## Оптимизация для Cloud.ru

### Использование нескольких GPU

Если доступно несколько GPU, обучение автоматически использует их через `device_map="auto"`.

### Экономия памяти

Для экономии памяти можно использовать:
- `--fp16`: Использование float16 вместо float32
- `--gradient-checkpointing`: Экономия памяти за счет скорости
- `--max-samples`: Ограничение размера датасета

### Сохранение чекпоинтов

Модель автоматически сохраняет чекпоинты каждые 100 шагов. Лучшая модель сохраняется в конце обучения.

## Troubleshooting

### Нехватка памяти GPU

Уменьшите `--batch-size` или увеличьте `--gradient-accumulation-steps`.

### Медленное обучение

- Увеличьте `--batch-size` если есть свободная память
- Уменьшите `--max-length`
- Используйте `--max-samples` для теста

### Ошибки загрузки модели

Убедитесь, что модель доступна на Hugging Face или загружена локально.

