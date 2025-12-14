"""
Демонстрационный скрипт для показа работы HR-агента
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.hr_vacancies_agent import HRVacanciesAgent


async def demo():
    """Демонстрация работы агента"""
    print("=" * 70)
    print("HR-АГЕНТ ДЛЯ АНАЛИЗА ВАКАНСИЙ - ДЕМОНСТРАЦИЯ")
    print("=" * 70)
    print()
    
    # Проверка API ключа
    if not os.getenv("EVOLUTION_API_KEY") or os.getenv("EVOLUTION_API_KEY") == "your_api_key_here":
        print("⚠️  ВНИМАНИЕ: API ключ не настроен!")
        print("Создайте файл .env на основе .env.example и укажите ваш EVOLUTION_API_KEY")
        print()
        return
    
    agent = HRVacanciesAgent()
    
    # Примеры запросов
    demo_queries = [
        {
            "query": "Найди 5 вакансий Python разработчика",
            "description": "Поиск вакансий по ключевому слову"
        },
        {
            "query": "Покажи статистику по вакансиям программиста",
            "description": "Получение статистики по рынку"
        },
        {
            "query": "Найди вакансии с зарплатой от 150000 до 250000 рублей в Москве",
            "description": "Поиск с фильтрами по зарплате и региону"
        }
    ]
    
    for i, demo_item in enumerate(demo_queries, 1):
        print(f"\n{'='*70}")
        print(f"ДЕМО {i}/{len(demo_queries)}: {demo_item['description']}")
        print(f"{'='*70}")
        print(f"Запрос пользователя: {demo_item['query']}")
        print("-" * 70)
        print("Обработка...")
        print()
        
        try:
            result = await agent.process_query(demo_item['query'])
            print("Ответ агента:")
            print(result)
            
            # Показываем A2A формат
            a2a_response = agent.format_a2a_response(result)
            print("\n" + "-" * 70)
            print("Формат ответа (A2A):")
            import json
            print(json.dumps(a2a_response, ensure_ascii=False, indent=2))
            
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 70)
    print("Демонстрация завершена!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())

