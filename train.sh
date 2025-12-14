#!/bin/bash

# Скрипт для запуска обучения модели на Cloud.ru
# Использование: ./train.sh

echo "=========================================="
echo "ОБУЧЕНИЕ МОДЕЛИ ДЛЯ АНАЛИЗА ВАКАНСИЙ"
echo "=========================================="
echo ""

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "❌ Виртуальное окружение не найдено!"
    echo "Создайте его: python3 -m venv venv"
    exit 1
fi

# Активация виртуального окружения
echo "Активация виртуального окружения..."
source venv/bin/activate

# Проверка GPU
echo ""
echo "Проверка GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')" || {
    echo "❌ Ошибка проверки GPU"
    exit 1
}

echo ""
echo "Начало обучения..."
echo ""

# Обучение модели
python train_model.py \
    --model-name IlyaGusev/saiga_mistral_7b_merged \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --hf-split train \
    --output-dir ./models/finetuned \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --max-length 512

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!"
    echo "=========================================="
    echo "Модель сохранена в: ./models/finetuned"
else
    echo ""
    echo "=========================================="
    echo "❌ ОШИБКА ПРИ ОБУЧЕНИИ"
    echo "=========================================="
    exit 1
fi

