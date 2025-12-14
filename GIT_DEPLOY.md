# Инструкция по деплою в Git

## Инициализация репозитория

### 1. Инициализация Git

```bash
git init
```

### 2. Добавление всех файлов

```bash
git add .
```

### 3. Первый коммит

```bash
git commit -m "Initial commit: HR-агент для анализа вакансий с MCP-сервером"
```

## Настройка удаленного репозитория

### GitHub

1. Создайте новый репозиторий на GitHub
2. Добавьте remote:

```bash
git remote add origin https://github.com/ваш-username/resume-ml.git
```

3. Переименуйте ветку в main (если нужно):

```bash
git branch -M main
```

4. Отправьте код:

```bash
git push -u origin main
```

### GitLab

1. Создайте новый проект на GitLab
2. Добавьте remote:

```bash
git remote add origin https://gitlab.com/ваш-username/resume-ml.git
```

3. Отправьте код:

```bash
git push -u origin main
```

### Cloud.ru Git

1. Создайте репозиторий в Cloud.ru
2. Добавьте remote:

```bash
git remote add origin <URL_репозитория_Cloud.ru>
```

3. Отправьте код:

```bash
git push -u origin main
```

## Структура коммитов

Рекомендуется делать осмысленные коммиты:

```bash
# Основной функционал
git commit -m "feat: добавлен MCP-сервер для работы с вакансиями"
git commit -m "feat: добавлен AI-агент с интеграцией Evolution Foundation Models"

# Конфигурация
git commit -m "config: добавлены requirements.txt и .env.example"
git commit -m "config: добавлен tools.json для платформы"

# Документация
git commit -m "docs: добавлен README с полной документацией"
git commit -m "docs: добавлены инструкции по обучению модели"

# Тесты
git commit -m "test: добавлены тесты для MCP-сервера и агента"
```

## Игнорируемые файлы

Файл `.gitignore` уже настроен и исключает:
- `.env` (секретные ключи)
- `__pycache__/` (кэш Python)
- `*.csv` (большие файлы данных)
- `models/` (обученные модели)
- IDE файлы

## Работа с большими файлами

### CSV файл с вакансиями

CSV файл (`IT_vacancies_full 2.csv`) исключен из Git через `.gitignore`, так как он слишком большой (180MB).

Если нужно добавить данные:
1. Используйте Git LFS:
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
git add "IT_vacancies_full 2.csv"
```

2. Или загрузите на облачное хранилище и укажите URL в документации

### Обученные модели

Модели также исключены. После обучения на Cloud.ru:
1. Сохраните модель в облачное хранилище
2. Или используйте Git LFS для моделей
3. Или добавьте инструкции по скачиванию модели

## CI/CD (опционально)

### GitHub Actions

Создайте `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python test_mcp.py
      - run: python test_agent.py
```

## Обновление репозитория

После изменений:

```bash
# Проверка статуса
git status

# Добавление изменений
git add .

# Коммит
git commit -m "Описание изменений"

# Отправка
git push
```

## Клонирование репозитория

После публикации, другие могут клонировать:

```bash
git clone https://github.com/ваш-username/resume-ml.git
cd resume-ml
pip install -r requirements.txt
cp .env.example .env
# Отредактируйте .env
```

## Публикация релизов

Создайте теги для версий:

```bash
git tag -a v1.0.0 -m "Первая версия HR-агента"
git push origin v1.0.0
```

