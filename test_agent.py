"""
Тестовый скрипт для проверки работы HR-агента
"""

import asyncio
import os
from dotenv import load_dotenv
from agent.hr_vacancies_agent import HRVacanciesAgent

# Загрузка переменных окружения
load_dotenv()


async def test_agent():
    """Тестирование основных функций агента"""
    agent = HRVacanciesAgent()
    
    test_queries = [
        "Найди вакансии Python разработчика",
        "Покажи статистику по вакансиям программиста",
        "Найди вакансии с зарплатой от 150000 до 250000 рублей в Москве"
    ]
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ HR-АГЕНТА ДЛЯ ВАКАНСИЙ")
    print("=" * 60)
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"Тест {i}/{len(test_queries)}")
        print(f"Запрос: {query}")
        print("-" * 60)
        
        try:
            result = await agent.process_query(query)
            print(f"Ответ: {result}")
        except Exception as e:
            print(f"Ошибка: {str(e)}")
        
        print()
        print("=" * 60)
        print()


if __name__ == "__main__":
    # Проверка наличия API ключа
    if not os.getenv("EVOLUTION_API_KEY"):
        print("ВНИМАНИЕ: EVOLUTION_API_KEY не установлен в .env файле")
        print("Создайте .env файл на основе .env.example и укажите ваш API ключ")
        exit(1)
    
    asyncio.run(test_agent())

