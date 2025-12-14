# Руководство по использованию Hugging Face датасета

## Датасет: evilfreelancer/headhunter

Этот проект поддерживает загрузку данных напрямую с Hugging Face, что упрощает процесс обучения модели.

## Быстрый старт

### 1. Проверка датасета

```bash
# Посмотреть доступные splits
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action list

# Посмотреть образцы данных (первые 10 примеров)
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action sample

# Загрузить и проверить структуру датасета
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action load --num-samples 100
```

### 2. Обучение модели

```bash
# Обучение с использованием Hugging Face датасета
python train_model.py \
    --model-name IlyaGusev/saiga_mistral_7b_merged \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --hf-split train \
    --output-dir ./models/finetuned \
    --num-epochs 3 \
    --batch-size 4 \
    --max-samples 1000
```

## API Hugging Face Datasets Server

Проект использует официальный API Hugging Face для загрузки данных:

### Получение строк данных

```bash
curl -X GET \
     "https://datasets-server.huggingface.co/rows?dataset=evilfreelancer%2Fheadhunter&config=default&split=train&offset=0&length=100"
```

### Список доступных splits

```bash
curl -X GET \
     "https://datasets-server.huggingface.co/splits?dataset=evilfreelancer%2Fheadhunter"
```

### Список Parquet файлов

```bash
curl -X GET \
     "https://huggingface.co/api/datasets/evilfreelancer/headhunter/parquet/default/train"
```

## Преимущества использования Hugging Face

1. **Не нужно загружать большие файлы** - данные загружаются автоматически
2. **Актуальные данные** - всегда используется последняя версия датасета
3. **Кэширование** - библиотека `datasets` автоматически кэширует данные
4. **Гибкость** - легко переключаться между разными датасетами

## Параметры загрузки

- `--use-hf`: Флаг для использования Hugging Face вместо CSV
- `--hf-dataset`: Название датасета (например: `evilfreelancer/headhunter`)
- `--hf-split`: Split для загрузки (обычно `train`, `test`, `validation`)
- `--max-samples`: Ограничение количества примеров (для быстрого теста)

## Примеры использования

### Обучение на полном датасете

```bash
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --num-epochs 5 \
    --batch-size 8
```

### Быстрый тест на небольшой выборке

```bash
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --max-samples 100 \
    --num-epochs 1 \
    --batch-size 4
```

### Использование другого split

```bash
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --hf-split test \
    --max-samples 500
```

## Troubleshooting

### Ошибка: "Dataset not found"

Убедитесь, что название датасета указано правильно:
- Формат: `username/dataset-name`
- Пример: `evilfreelancer/headhunter`

### Ошибка: "Split not found"

Проверьте доступные splits:
```bash
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action list
```

### Медленная загрузка

Используйте `--max-samples` для ограничения размера датасета при тестировании. При первом запуске данные будут закэшированы.

### Проблемы с памятью

Если датасет слишком большой:
1. Используйте `--max-samples` для ограничения
2. Уменьшите `--batch-size`
3. Используйте streaming загрузку (требует дополнительной настройки)

## Дополнительная информация

- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Datasets Server API](https://huggingface.co/docs/datasets-server/)
- Датасет: https://huggingface.co/datasets/evilfreelancer/headhunter

