# Пошаговая инструкция: Обучение модели на Cloud.ru

## 🎯 Цель
Обучить модель на датасете Hugging Face `evilfreelancer/headhunter` используя GPU мощности Cloud.ru.

---

## 📋 Шаг 1: Подготовка на Cloud.ru

### 1.1. Создайте GPU инстанс

1. Войдите в панель Cloud.ru
2. Создайте виртуальную машину с GPU:
   - **ОС**: Ubuntu 22.04 или 20.04
   - **GPU**: Выберите GPU с минимум 16GB памяти (например, NVIDIA A100, V100)
   - **RAM**: Минимум 32GB
   - **Диск**: Минимум 100GB

### 1.2. Подключитесь к инстансу

```bash
ssh ваш_пользователь@ip_адрес_инстанса
```

---

## 📦 Шаг 2: Установка зависимостей

### 2.1. Обновите систему

```bash
sudo apt update
sudo apt upgrade -y
```

### 2.2. Установите Python и pip

```bash
sudo apt install -y python3.10 python3-pip python3-venv
```

### 2.3. Установите CUDA (если не установлена)

Проверьте наличие CUDA:
```bash
nvidia-smi
```

Если CUDA не установлена, следуйте инструкциям Cloud.ru для установки CUDA на вашем инстансе.

### 2.4. Клонируйте репозиторий

```bash
# Если код уже в Git
git clone https://github.com/Human221/resume-ml.git
cd resume-ml

# Или загрузите файлы через scp с вашего компьютера:
# scp -r /Users/rustam/Desktop/resume-ml ваш_пользователь@ip_адрес:/home/ваш_пользователь/
```

### 2.5. Создайте виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.6. Установите зависимости

```bash
# Установите PyTorch с поддержкой CUDA (важно!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установите остальные зависимости
pip install -r requirements-train.txt
```

**Проверка установки:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

---

## 🔍 Шаг 3: Проверка датасета

### 3.1. Проверьте доступность датасета

```bash
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action list
```

Должны увидеть список доступных splits (train, test, validation и т.д.)

### 3.2. Посмотрите образцы данных

```bash
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action sample --num-samples 5
```

Это покажет структуру данных и примеры вакансий.

---

## 🚀 Шаг 4: Обучение модели

### 4.1. Быстрый тест (рекомендуется сначала)

Обучите модель на небольшой выборке для проверки:

```bash
python train_model.py \
    --model-name IlyaGusev/saiga_mistral_7b_merged \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --hf-split train \
    --output-dir ./models/finetuned \
    --num-epochs 1 \
    --batch-size 2 \
    --max-samples 100 \
    --learning-rate 2e-5
```

**Параметры для теста:**
- `--max-samples 100` - только 100 примеров (быстро)
- `--num-epochs 1` - одна эпоха
- `--batch-size 2` - маленький батч (безопасно для памяти)

### 4.2. Полное обучение

После успешного теста запустите полное обучение:

```bash
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
```

**Параметры для полного обучения:**
- Убрано `--max-samples` - используется весь датасет
- `--num-epochs 3` - 3 эпохи
- `--batch-size 4` - можно увеличить до 8, если есть память

### 4.3. Обучение в фоновом режиме (для длительных процессов)

Если обучение займет много времени, запустите в фоне:

```bash
# Запуск в screen (рекомендуется)
screen -S training
python train_model.py \
    --model-name IlyaGusev/saiga_mistral_7b_merged \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --output-dir ./models/finetuned \
    --num-epochs 3 \
    --batch-size 4

# Отключиться от screen: Ctrl+A, затем D
# Вернуться: screen -r training
```

Или используйте `nohup`:

```bash
nohup python train_model.py \
    --model-name IlyaGusev/saiga_mistral_7b_merged \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --output-dir ./models/finetuned \
    --num-epochs 3 \
    --batch-size 4 \
    > training.log 2>&1 &

# Проверка прогресса
tail -f training.log
```

---

## 📊 Шаг 5: Мониторинг обучения

### 5.1. Проверка использования GPU

В другом терминале:
```bash
watch -n 1 nvidia-smi
```

### 5.2. Проверка логов

```bash
# Если используете nohup
tail -f training.log

# Если используете screen
screen -r training
```

### 5.3. Проверка сохраненных чекпоинтов

```bash
ls -lh ./models/finetuned/checkpoint-*/
```

---

## ✅ Шаг 6: Проверка результата

После завершения обучения:

### 6.1. Проверьте сохраненную модель

```bash
ls -lh ./models/finetuned/
```

Должны быть файлы:
- `config.json`
- `pytorch_model.bin` или `model.safetensors`
- `tokenizer.json`
- и другие файлы токенизатора

### 6.2. Протестируйте модель (опционально)

Создайте простой тестовый скрипт:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./models/finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Тестовый промпт
prompt = "<|im_start|>system\nТы - HR-ассистент.<|im_end|>\n<|im_start|>user\nОпиши вакансию Python разработчика<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

---

## 💾 Шаг 7: Сохранение модели

### 7.1. Скачайте модель на локальный компьютер

```bash
# С вашего компьютера
scp -r ваш_пользователь@ip_адрес:/home/ваш_пользователь/resume-ml/models/finetuned ./models/
```

### 7.2. Или загрузите в облачное хранилище Cloud.ru

Следуйте инструкциям Cloud.ru для загрузки файлов в Object Storage.

---

## 🔧 Настройка параметров под ваш GPU

### Для GPU с 16GB памяти (например, NVIDIA T4, RTX 3090)

```bash
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --num-epochs 3
```

### Для GPU с 24GB+ памяти (например, NVIDIA A100, RTX 4090)

```bash
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --num-epochs 3
```

### Для GPU с 40GB+ памяти (например, NVIDIA A100 40GB)

```bash
python train_model.py \
    --use-hf \
    --hf-dataset evilfreelancer/headhunter \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --num-epochs 3
```

---

## ⚠️ Troubleshooting

### Ошибка: "Out of memory"

Уменьшите `--batch-size`:
```bash
--batch-size 1  # вместо 4
```

Или увеличьте `--gradient-accumulation-steps` в коде (нужно будет отредактировать `train_model.py`).

### Ошибка: "CUDA not available"

Проверьте:
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

Убедитесь, что установлен PyTorch с CUDA поддержкой.

### Ошибка: "Dataset not found"

Проверьте интернет-соединение и доступность Hugging Face:
```bash
curl https://huggingface.co/datasets/evilfreelancer/headhunter
```

### Медленное обучение

- Увеличьте `--batch-size` если есть свободная память
- Уменьшите `--max-length` (например, до 256)
- Используйте `--max-samples` для теста на меньшем датасете

---

## 📝 Пример полного скрипта запуска

Создайте файл `train.sh`:

```bash
#!/bin/bash

# Активация виртуального окружения
source venv/bin/activate

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

echo "Обучение завершено!"
```

Сделайте исполняемым и запустите:
```bash
chmod +x train.sh
./train.sh
```

---

## 🎓 Дополнительные ресурсы

- [TRAINING.md](TRAINING.md) - Общая документация по обучению
- [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md) - Работа с Hugging Face
- [README.md](README.md) - Общая документация проекта

---

## ⏱️ Оценка времени обучения

- **Тест (100 примеров, 1 эпоха)**: ~10-30 минут
- **Полное обучение (весь датасет, 3 эпохи)**: 
  - На GPU 16GB: ~4-8 часов
  - На GPU 24GB: ~2-4 часа
  - На GPU 40GB+: ~1-2 часа

*Время зависит от размера датасета и мощности GPU*

