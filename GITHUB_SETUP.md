# Инструкция по загрузке кода на GitHub

## Шаг 1: Создайте репозиторий на GitHub

1. Перейдите на https://github.com/new
2. Заполните форму:
   - **Repository name**: `resume-ml`
   - **Description**: `HR-агент для анализа вакансий с MCP-сервером`
   - **Visibility**: Public или Private (на ваше усмотрение)
   - **НЕ** ставьте галочки на "Initialize this repository with a README" (у нас уже есть файлы)
3. Нажмите **Create repository**

## Шаг 2: Настройте аутентификацию

GitHub больше не поддерживает пароли. Нужен Personal Access Token (PAT).

### Вариант A: Personal Access Token (рекомендуется)

1. Перейдите в **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
   Или напрямую: https://github.com/settings/tokens

2. Нажмите **Generate new token** → **Generate new token (classic)**

3. Настройте токен:
   - **Note**: `resume-ml deployment`
   - **Expiration**: выберите срок действия
   - **Scopes**: отметьте `repo` (полный доступ к репозиториям)

4. Нажмите **Generate token**

5. **ВАЖНО**: Скопируйте токен сразу (он показывается только один раз!)

6. Используйте токен вместо пароля при push:
```bash
# При запросе пароля введите токен
git push -u origin main
```

### Вариант B: SSH ключ (альтернатива)

1. Проверьте наличие SSH ключа:
```bash
ls -al ~/.ssh
```

2. Если ключа нет, создайте:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

3. Добавьте ключ в ssh-agent:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

4. Скопируйте публичный ключ:
```bash
cat ~/.ssh/id_ed25519.pub
```

5. Добавьте ключ на GitHub:
   - Settings → SSH and GPG keys → New SSH key
   - Вставьте скопированный ключ

6. Измените remote на SSH:
```bash
git remote set-url origin git@github.com:Human221/resume-ml.git
```

## Шаг 3: Отправьте код

После настройки аутентификации:

```bash
# Проверьте, что remote настроен
git remote -v

# Отправьте код
git push -u origin main
```

## Шаг 4: Проверьте результат

Перейдите на https://github.com/Human221/resume-ml и убедитесь, что все файлы загружены.

## Быстрая команда (если репозиторий уже создан)

Если репозиторий уже создан на GitHub, просто выполните:

```bash
# Убедитесь, что вы в правильной директории
cd /Users/rustam/Desktop/resume-ml

# Проверьте статус
git status

# Отправьте код (потребуется токен при первом push)
git push -u origin main
```

## Troubleshooting

### Ошибка: "repository not found"
- Убедитесь, что репозиторий создан на GitHub
- Проверьте правильность имени: `resume-ml`
- Проверьте права доступа к репозиторию

### Ошибка: "Authentication failed"
- Используйте Personal Access Token вместо пароля
- Или настройте SSH ключ

### Ошибка: "remote origin already exists"
Если remote уже добавлен, но с неправильным URL:
```bash
git remote remove origin
git remote add origin https://github.com/Human221/resume-ml.git
```

## После успешной загрузки

Ваш код будет доступен по адресу:
**https://github.com/Human221/resume-ml**

Другие смогут клонировать репозиторий:
```bash
git clone https://github.com/Human221/resume-ml.git
```

