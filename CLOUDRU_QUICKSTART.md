# Быстрый старт: Обучение на Cloud.ru

## ✅ Подтверждение конфигурации

Ваша конфигурация отлично подходит для обучения:

- ✅ **vCPU: 8** - достаточно для обработки данных
- ✅ **RAM: 64 ГБ** - отлично для больших моделей
- ✅ **Ubuntu 24.04** - современная версия с Python 3.12
- ✅ **256 ГБ диск** - достаточно для модели и данных
- ✅ **100% vCPU** - гарантированная производительность

**Подтверждаю выбор!** 🚀

## 🚀 Быстрая инструкция (после создания VM)

### 1. Подключитесь к серверу

```bash
ssh ваш_пользователь@ip_адрес_сервера
```

### 2. Клонируйте репозиторий

```bash
git clone https://github.com/Human221/resume-ml.git
cd resume-ml
```

### 3. Создайте виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 4. Установите PyTorch с CUDA

```bash
# Для CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Или для CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Установите остальные зависимости

```bash
pip install -r requirements-train.txt
```

### 6. Проверьте GPU

```bash
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

### 7. Проверьте датасет

```bash
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action list
```

### 8. Запустите обучение

**Быстрый тест (10-30 минут):**
```bash
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --max-samples 100 \
    --num-epochs 1 \
    --batch-size 2
```

**Полное обучение (2-4 часа):**
```bash
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --output-dir ./models/finetuned \
    --num-epochs 3 \
    --batch-size 4
```

Или используйте готовый скрипт:
```bash
chmod +x train.sh
./train.sh
```

## 📊 Мониторинг

В другом терминале:
```bash
watch -n 1 nvidia-smi
```

## ⚠️ Важно

- На Mac можно только подготовить код, обучение - только на Cloud.ru
- Убедитесь, что на сервере установлен NVIDIA драйвер и CUDA
- Первое обучение может занять больше времени (загрузка модели и данных)

## 📚 Подробная инструкция

См. [CLOUDRU_TRAINING.md](CLOUDRU_TRAINING.md) для детальной инструкции.

