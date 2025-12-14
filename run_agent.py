#!/usr/bin/env python3
"""
Интерактивный запуск HR-агента
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.hr_vacancies_agent import HRVacanciesAgent


async def interactive_mode():
    """Интерактивный режим работы с агентом"""
    print("=" * 70)
    print("HR-АГЕНТ ДЛЯ АНАЛИЗА ВАКАНСИЙ")
    print("=" * 70)
    print()
    print("Введите ваш запрос на естественном языке.")
    print("Примеры:")
    print("  - Найди вакансии Python разработчика")
    print("  - Покажи статистику по вакансиям программиста")
    print("  - Найди вакансии с зарплатой от 150000 рублей")
    print()
    print("Для выхода введите 'exit' или 'quit'")
    print("=" * 70)
    print()
    
    if not os.getenv("EVOLUTION_API_KEY") or os.getenv("EVOLUTION_API_KEY") == "your_api_key_here":
        print("⚠️  ВНИМАНИЕ: API ключ не настроен!")
        print("Создайте файл .env на основе .env.example и укажите ваш EVOLUTION_API_KEY")
        print()
        return
    
    agent = HRVacanciesAgent()
    conversation_history = []
    
    while True:
        try:
            user_input = input("\nВы: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("\nДо свидания!")
                break
            
            if not user_input:
                continue
            
            print("\nОбработка...")
            result = await agent.process_query(user_input, conversation_history)
            
            print(f"\nАгент: {result}")
            
            # Обновляем историю
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": result})
            
            # Ограничиваем размер истории
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(interactive_mode())

