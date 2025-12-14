# Настройка на Mac для разработки

## ⚠️ Важно

**На Mac можно только разрабатывать и тестировать код. Обучение модели нужно делать на Cloud.ru с GPU!**

## Проблемы на Mac

1. **Python версия**: У вас Python 3.9, а некоторые пакеты требуют 3.10+
2. **CUDA**: На Mac нет NVIDIA GPU, поэтому PyTorch с CUDA не нужен
3. **nvidia-smi**: Эта команда работает только на Linux с NVIDIA GPU

## Решение: Разделение окружений

### Для разработки на Mac

Используйте `requirements-dev.txt` (без GPU зависимостей):

```bash
# Обновите pip
python3 -m pip install --upgrade pip

# Установите зависимости для разработки
pip install -r requirements-dev.txt
```

### Для обучения на Cloud.ru

Используйте `requirements-train.txt` на сервере с GPU.

## Проверка датасета на Mac

Вы можете проверить датасет без GPU:

```bash
# Установите зависимости для работы с датасетами
pip install datasets requests

# Проверьте датасет
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action list
python download_hf_dataset.py --dataset evilfreelancer/headhunter --action sample
```

## Обновление Python на Mac (опционально)

Если хотите использовать все функции на Mac, обновите Python до 3.10+:

### Через Homebrew

```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Или используйте pyenv

```bash
brew install pyenv
pyenv install 3.11.0
pyenv local 3.11.0
python -m venv venv
source venv/bin/activate
```

## Что можно делать на Mac

✅ Разрабатывать и тестировать код  
✅ Проверять датасеты  
✅ Тестировать MCP-сервер (если установлен Python 3.10+)  
✅ Тестировать агента (без обучения)  
✅ Подготавливать код для деплоя  

❌ Обучать модель (нужен GPU на Cloud.ru)  
❌ Использовать CUDA  

## Рекомендация

**Для обучения модели:**
1. Подготовьте код на Mac
2. Загрузите на Cloud.ru через Git
3. Установите зависимости на Cloud.ru
4. Запустите обучение на GPU

См. [CLOUDRU_TRAINING.md](CLOUDRU_TRAINING.md) для инструкций по обучению на Cloud.ru.

