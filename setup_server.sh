#!/bin/bash

# Автоматическая установка и настройка на сервере Cloud.ru
# Использование: ./setup_server.sh

set -e  # Остановка при ошибке

echo "=========================================="
echo "НАСТРОЙКА СЕРВЕРА CLOUD.RU"
echo "=========================================="
echo ""

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Проверка, что мы на Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}ОШИБКА: Этот скрипт нужно запускать на Linux сервере!${NC}"
    echo "Вы находитесь на: $OSTYPE"
    exit 1
fi

echo -e "${GREEN}✓ Проверка системы...${NC}"
echo "ОС: $(lsb_release -d | cut -f2)"
echo "Python: $(python3 --version)"
echo ""

# Шаг 1: Обновление системы
echo -e "${YELLOW}[1/8] Обновление системы...${NC}"
sudo apt update
sudo apt upgrade -y

# Шаг 2: Установка базовых пакетов
echo -e "${YELLOW}[2/8] Установка базовых пакетов...${NC}"
sudo apt install -y python3-pip python3-venv git curl wget

# Шаг 3: Проверка GPU
echo -e "${YELLOW}[3/8] Проверка GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA драйвер установлен${NC}"
    nvidia-smi
else
    echo -e "${YELLOW}⚠ NVIDIA драйвер не найден. Установка...${NC}"
    sudo ubuntu-drivers autoinstall
    echo -e "${YELLOW}⚠ Может потребоваться перезагрузка. Выполните: sudo reboot${NC}"
fi

# Шаг 4: Клонирование репозитория
echo -e "${YELLOW}[4/8] Клонирование репозитория...${NC}"
if [ -d "resume-ml" ]; then
    echo "Директория уже существует, обновление..."
    cd resume-ml
    git pull
else
    cd ~
    git clone https://github.com/Human221/resume-ml.git
    cd resume-ml
fi

# Шаг 5: Создание виртуального окружения
echo -e "${YELLOW}[5/8] Создание виртуального окружения...${NC}"
if [ -d "venv" ]; then
    echo "Виртуальное окружение уже существует"
else
    python3 -m venv venv
fi

source venv/bin/activate

# Шаг 6: Обновление pip
echo -e "${YELLOW}[6/8] Обновление pip...${NC}"
pip install --upgrade pip

# Шаг 7: Установка PyTorch
echo -e "${YELLOW}[7/8] Установка PyTorch с CUDA...${NC}"
echo "Это может занять несколько минут..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Шаг 8: Установка остальных зависимостей
echo -e "${YELLOW}[8/8] Установка зависимостей...${NC}"
pip install -r requirements-train.txt

# Проверка установки
echo ""
echo "=========================================="
echo "ПРОВЕРКА УСТАНОВКИ"
echo "=========================================="

echo -e "${YELLOW}Проверка PyTorch и CUDA...${NC}"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('⚠ CUDA недоступна')
"

echo ""
echo -e "${YELLOW}Проверка других библиотек...${NC}"
python3 -c "
try:
    import transformers
    import datasets
    print('✓ transformers установлен')
    print('✓ datasets установлен')
except ImportError as e:
    print(f'✗ Ошибка: {e}')
"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ УСТАНОВКА ЗАВЕРШЕНА!${NC}"
echo "=========================================="
echo ""
echo "Следующие шаги:"
echo "1. Проверьте датасет:"
echo "   python download_hf_dataset.py --dataset evilfreelancer/headhunter --action list"
echo ""
echo "2. Запустите обучение:"
echo "   python train_model.py --use-hf --hf-dataset evilfreelancer/headhunter --max-samples 100 --num-epochs 1 --batch-size 2"
echo ""

