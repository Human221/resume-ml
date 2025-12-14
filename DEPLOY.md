# Инструкция по деплою в Cloud.ru Evolution AI Agents

## Требования для деплоя

1. **tools.json** - ✅ Создан (описание всех инструментов MCP)
2. **.env** - ✅ Шаблон создан (.env.example)
3. **Совместимость с A2A протоколом** - ✅ Реализована в `agent/hr_vacancies_agent.py`
4. **MCP-протокол** - ✅ Реализован через FastMCP

## Структура для деплоя

```
resume-ml/
├── mcp_server/
│   ├── __init__.py
│   └── vacancies_mcp.py          # MCP-сервер
├── agent/
│   ├── __init__.py
│   └── hr_vacancies_agent.py     # AI-агент
├── requirements.txt               # Зависимости
├── .env.example                   # Шаблон конфигурации
├── tools.json                     # Описание инструментов
├── README.md                      # Документация
└── run_agent.py                   # Точка входа для агента
```

## Переменные окружения

Убедитесь, что в `.env` указаны:

```env
EVOLUTION_API_KEY=ваш_ключ
EVOLUTION_API_BASE=https://foundation-models.api.cloud.ru/v1/
EVOLUTION_MODEL=gpt-4o-mini
MCP_SERVER_PATH=python
MCP_SCRIPT_PATH=mcp_server/vacancies_mcp.py
VACANCIES_CSV_PATH=IT_vacancies_full 2.csv
```

## Формат ответа A2A

Агент возвращает ответы в формате:

```json
{
  "response": {
    "content": "Текст ответа",
    "type": "text"
  },
  "metadata": {},
  "status": "success"
}
```

## Проверка перед деплоем

1. ✅ Все зависимости в `requirements.txt`
2. ✅ MCP-сервер работает (`python test_mcp.py`)
3. ✅ Агент работает (`python test_agent.py`)
4. ✅ `.env` настроен корректно
5. ✅ `tools.json` содержит все инструменты

## Тестирование на платформе

После деплоя проверьте:

1. Загрузку MCP-сервера
2. Доступность инструментов через платформу
3. Работу агента с Evolution Foundation Models
4. Корректность формата ответов A2A

