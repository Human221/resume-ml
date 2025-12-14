"""
Тестовый скрипт для проверки работы MCP-сервера
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Тестирование MCP-сервера напрямую"""
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server/vacancies_mcp.py"],
        env=None
    )
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ MCP-СЕРВЕРА")
    print("=" * 60)
    print()
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Получаем список инструментов
            print("1. Получение списка инструментов...")
            tools_result = await session.list_tools()
            print(f"   Найдено инструментов: {len(tools_result.tools)}")
            for tool in tools_result.tools:
                print(f"   - {tool.name}: {tool.description}")
            print()
            
            # Тест 1: Поиск вакансий
            print("2. Тест поиска вакансий...")
            try:
                result = await session.call_tool(
                    "search_vacancies",
                    {
                        "query": "Python",
                        "limit": 3
                    }
                )
                print(f"   Результат: {result.content[0].text[:200]}...")
            except Exception as e:
                print(f"   Ошибка: {str(e)}")
            print()
            
            # Тест 2: Получение статистики
            print("3. Тест получения статистики...")
            try:
                result = await session.call_tool(
                    "get_vacancy_statistics",
                    {}
                )
                print(f"   Результат: {result.content[0].text[:200]}...")
            except Exception as e:
                print(f"   Ошибка: {str(e)}")
            print()
            
            print("=" * 60)
            print("Тестирование завершено")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())

