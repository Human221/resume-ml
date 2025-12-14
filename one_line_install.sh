#!/bin/bash
# Одна команда для установки и запуска (выполнять НА СЕРВЕРЕ)

cd ~ && \
git clone https://github.com/Human221/resume-ml.git && \
cd resume-ml && \
chmod +x setup_server.sh && \
./setup_server.sh && \
source venv/bin/activate && \
echo "==========================================" && \
echo "Установка завершена! Запуск обучения..." && \
echo "==========================================" && \
python train_model.py --use-hf --hf-dataset evilfreelancer/headhunter --max-samples 100 --num-epochs 1 --batch-size 2 --output-dir ./models/finetuned
