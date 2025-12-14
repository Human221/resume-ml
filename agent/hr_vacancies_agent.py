"""
HR-агент для работы с вакансиями
Использует Evolution Foundation Models и MCP-сервер для анализа вакансий
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class HRVacanciesAgent:
    """AI-агент для работы с вакансиями через MCP"""
    
    def __init__(self):
        self.api_key = os.getenv("EVOLUTION_API_KEY", "")
        self.api_base = os.getenv("EVOLUTION_API_BASE", "https://foundation-models.api.cloud.ru/v1/")
        self.model = os.getenv("EVOLUTION_MODEL", "gpt-4o-mini")
        self.mcp_server_path = os.getenv("MCP_SERVER_PATH", "python")
        self.mcp_script_path = os.getenv("MCP_SCRIPT_PATH", "mcp_server/vacancies_mcp.py")
        
    async def call_llm(self, messages: list, tools: Optional[list] = None) -> Dict[str, Any]:
        """Вызов Evolution Foundation Models API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.api_base}chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return {
                    "error": f"Ошибка API: {str(e)}",
                    "response": None
                }
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Вызов инструмента MCP-сервера"""
        server_params = StdioServerParameters(
            command=self.mcp_server_path,
            args=[self.mcp_script_path],
            env=None
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Получаем список доступных инструментов
                tools_result = await session.list_tools()
                
                # Находим нужный инструмент
                tool = None
                for t in tools_result.tools:
                    if t.name == tool_name:
                        tool = t
                        break
                
                if not tool:
                    return json.dumps({
                        "status": "error",
                        "message": f"Инструмент {tool_name} не найден"
                    })
                
                # Вызываем инструмент
                result = await session.call_tool(tool_name, arguments)
                
                return result.content[0].text if result.content else ""
    
    def format_tools_for_llm(self) -> list:
        """Форматирование инструментов MCP для LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_vacancies",
                    "description": "Поиск вакансий по заданным критериям (название, зарплата, опыт, регион)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Поисковый запрос (название должности, ключевые слова)"
                            },
                            "min_salary": {
                                "type": "number",
                                "description": "Минимальная зарплата"
                            },
                            "max_salary": {
                                "type": "number",
                                "description": "Максимальная зарплата"
                            },
                            "experience": {
                                "type": "string",
                                "description": "Требуемый опыт работы"
                            },
                            "area": {
                                "type": "string",
                                "description": "Регион/город"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Максимальное количество результатов (1-100)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_vacancy_statistics",
                    "description": "Получение статистики по вакансиям (средняя зарплата, топ ролей, регионов)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Профессиональная роль для фильтрации"
                            },
                            "area": {
                                "type": "string",
                                "description": "Регион для фильтрации"
                            },
                            "specialization": {
                                "type": "string",
                                "description": "Специализация для фильтрации"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_vacancies",
                    "description": "Сравнение нескольких вакансий по основным параметрам",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vacancy_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Список ID вакансий для сравнения (2-5 вакансий)",
                                "minItems": 2,
                                "maxItems": 5
                            }
                        },
                        "required": ["vacancy_ids"]
                    }
                }
            }
        ]
    
    async def process_query(self, user_query: str, conversation_history: Optional[list] = None) -> str:
        """Обработка запроса пользователя"""
        if conversation_history is None:
            conversation_history = []
        
        # Системное сообщение
        system_message = {
            "role": "system",
            "content": """Ты - HR-ассистент, специализирующийся на анализе вакансий. 
Твоя задача - помогать пользователям находить подходящие вакансии, анализировать рынок труда,
сравнивать предложения и предоставлять статистику.

Используй доступные инструменты для:
1. Поиска вакансий по различным критериям
2. Получения статистики по рынку вакансий
3. Сравнения вакансий между собой

Отвечай на русском языке, будь дружелюбным и профессиональным.
Предоставляй структурированную информацию и конкретные рекомендации."""
        }
        
        messages = [system_message] + conversation_history + [
            {"role": "user", "content": user_query}
        ]
        
        tools = self.format_tools_for_llm()
        
        # Первый вызов LLM
        response = await self.call_llm(messages, tools)
        
        if "error" in response:
            return f"Ошибка: {response['error']}"
        
        assistant_message = response["choices"][0]["message"]
        messages.append(assistant_message)
        
        # Если LLM хочет вызвать инструмент
        if assistant_message.get("tool_calls"):
            for tool_call in assistant_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}
                
                # Вызываем MCP инструмент
                tool_result = await self.call_mcp_tool(tool_name, arguments)
                
                # Добавляем результат в контекст
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": tool_result
                })
            
            # Второй вызов LLM с результатами инструментов
            final_response = await self.call_llm(messages, tools)
            
            if "error" in final_response:
                return f"Ошибка: {final_response['error']}"
            
            return final_response["choices"][0]["message"]["content"]
        
        return assistant_message["content"]
    
    def format_a2a_response(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Форматирование ответа согласно протоколу A2A"""
        return {
            "response": {
                "content": content,
                "type": "text"
            },
            "metadata": metadata or {},
            "status": "success"
        }


async def main():
    """Пример использования агента"""
    agent = HRVacanciesAgent()
    
    # Пример запроса
    query = "Найди вакансии Python разработчика с зарплатой от 100000 до 200000 рублей"
    
    print(f"Запрос: {query}\n")
    print("Обработка...\n")
    
    result = await agent.process_query(query)
    
    print("Ответ агента:")
    print(result)
    
    # Форматированный ответ A2A
    a2a_response = agent.format_a2a_response(result)
    print("\n\nA2A формат:")
    print(json.dumps(a2a_response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

