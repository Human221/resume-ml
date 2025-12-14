#!/bin/bash
# Полный автоматический запуск: установка + обучение

echo "=========================================="
echo "ПОЛНАЯ АВТОМАТИЗАЦИЯ: УСТАНОВКА + ОБУЧЕНИЕ"
echo "=========================================="
echo ""

SERVER_IP="176.109.111.108"
SERVER_USER="root"

echo "Шаг 1: Отправка кода на GitHub..."
git add -A
git commit -m "Auto: обновление перед установкой" 2>/dev/null || true
git push origin main 2>/dev/null || echo "⚠️  Не удалось отправить на GitHub (возможно нужен токен)"

echo ""
echo "Шаг 2: Копирование файлов на сервер..."
echo "Введите пароль от сервера:"
scp setup_server.sh train.sh train_model.py download_hf_dataset.py requirements-train.txt $SERVER_USER@$SERVER_IP:~/

echo ""
echo "Шаг 3: Запуск установки на сервере..."
echo "Введите пароль от сервера еще раз:"
ssh $SERVER_USER@$SERVER_IP << 'ENDSSH'
cd ~
chmod +x setup_server.sh
./setup_server.sh

echo ""
echo "=========================================="
echo "УСТАНОВКА ЗАВЕРШЕНА. ЗАПУСК ОБУЧЕНИЯ..."
echo "=========================================="

cd ~/resume-ml || (git clone https://github.com/Human221/resume-ml.git && cd resume-ml)
source venv/bin/activate

echo "Проверка датасета..."
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action list

echo ""
echo "Запуск обучения (быстрый тест)..."
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --max-samples 100 \
    --num-epochs 1 \
    --batch-size 2 \
    --output-dir ./models/finetuned

echo ""
echo "=========================================="
echo "✅ ОБУЧЕНИЕ ЗАПУЩЕНО!"
echo "=========================================="
ENDSSH

echo ""
echo "Готово! Проверьте прогресс на сервере."
