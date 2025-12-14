"""
MCP-сервер для работы с вакансиями
Предоставляет инструменты для поиска, анализа и сравнения вакансий
"""

import os
import json
import csv
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from fastmcp import FastMCP
from pydantic import BaseModel, Field, model_validator

# Инициализация FastMCP
mcp = FastMCP("Vacancies MCP Server")


# Pydantic модели для валидации
class SearchVacanciesParams(BaseModel):
    """Параметры для поиска вакансий"""
    query: str = Field(..., description="Поисковый запрос (название должности, ключевые слова)")
    min_salary: Optional[float] = Field(None, description="Минимальная зарплата")
    max_salary: Optional[float] = Field(None, description="Максимальная зарплата")
    experience: Optional[str] = Field(None, description="Требуемый опыт работы")
    area: Optional[str] = Field(None, description="Регион/город")
    limit: int = Field(10, ge=1, le=100, description="Максимальное количество результатов")

    @model_validator(mode='after')
    def validate_salary_range(self):
        if self.max_salary and self.min_salary:
            if self.max_salary < self.min_salary:
                raise ValueError('max_salary должен быть больше min_salary')
        return self


class GetVacancyStatsParams(BaseModel):
    """Параметры для получения статистики по вакансиям"""
    role: Optional[str] = Field(None, description="Профессиональная роль для фильтрации")
    area: Optional[str] = Field(None, description="Регион для фильтрации")
    specialization: Optional[str] = Field(None, description="Специализация для фильтрации")


class CompareVacanciesParams(BaseModel):
    """Параметры для сравнения вакансий"""
    vacancy_ids: List[str] = Field(..., min_items=2, max_items=5, description="Список ID вакансий для сравнения")


class VacancyData:
    """Класс для работы с данными вакансий"""
    
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or os.getenv("VACANCIES_CSV_PATH", "IT_vacancies_full 2.csv")
        self.data: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self):
        """Загрузка данных из CSV"""
        try:
            csv_file = Path(self.csv_path)
            if not csv_file.exists():
                # Если файл не найден, создаем пустую структуру
                return
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            self.data = []
    
    def search(self, params: SearchVacanciesParams) -> List[Dict[str, Any]]:
        """Поиск вакансий по параметрам"""
        results = []
        query_lower = params.query.lower()
        
        for vacancy in self.data:
            # Поиск по названию и описанию
            name = vacancy.get('Name', '').lower()
            description = vacancy.get('Description', '').lower()
            keys = vacancy.get('Keys', '').lower()
            
            if query_lower not in name and query_lower not in description and query_lower not in keys:
                continue
            
            # Фильтр по зарплате
            if params.min_salary or params.max_salary:
                salary_from = vacancy.get('From', '')
                salary_to = vacancy.get('To', '')
                
                try:
                    from_val = float(salary_from) if salary_from else 0
                    to_val = float(salary_to) if salary_to else float('inf')
                    
                    if params.min_salary and to_val < params.min_salary:
                        continue
                    if params.max_salary and from_val > params.max_salary:
                        continue
                except (ValueError, TypeError):
                    pass
            
            # Фильтр по опыту
            if params.experience:
                exp = vacancy.get('Experience', '').lower()
                if params.experience.lower() not in exp:
                    continue
            
            # Фильтр по региону
            if params.area:
                area = vacancy.get('Area', '').lower()
                if params.area.lower() not in area:
                    continue
            
            results.append(vacancy)
            
            if len(results) >= params.limit:
                break
        
        return results
    
    def get_stats(self, params: GetVacancyStatsParams) -> Dict[str, Any]:
        """Получение статистики по вакансиям"""
        filtered = self.data
        
        if params.role:
            filtered = [v for v in filtered if params.role.lower() in v.get('Professional roles', '').lower()]
        
        if params.area:
            filtered = [v for v in filtered if params.area.lower() in v.get('Area', '').lower()]
        
        if params.specialization:
            filtered = [v for v in filtered if params.specialization.lower() in v.get('Specializations', '').lower()]
        
        if not filtered:
            return {
                "total": 0,
                "average_salary": 0,
                "salary_range": {"min": 0, "max": 0},
                "top_roles": [],
                "top_areas": []
            }
        
        # Расчет статистики по зарплатам
        salaries = []
        for v in filtered:
            try:
                from_val = float(v.get('From', 0) or 0)
                to_val = float(v.get('To', 0) or 0)
                if from_val > 0:
                    salaries.append(from_val)
                if to_val > 0:
                    salaries.append(to_val)
            except (ValueError, TypeError):
                continue
        
        avg_salary = sum(salaries) / len(salaries) if salaries else 0
        min_salary = min(salaries) if salaries else 0
        max_salary = max(salaries) if salaries else 0
        
        # Топ ролей
        roles = {}
        for v in filtered:
            role = v.get('Professional roles', '')
            if role:
                roles[role] = roles.get(role, 0) + 1
        
        top_roles = sorted(roles.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Топ регионов
        areas = {}
        for v in filtered:
            area = v.get('Area', '')
            if area:
                areas[area] = areas.get(area, 0) + 1
        
        top_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total": len(filtered),
            "average_salary": round(avg_salary, 2),
            "salary_range": {"min": round(min_salary, 2), "max": round(max_salary, 2)},
            "top_roles": [{"role": r[0], "count": r[1]} for r in top_roles],
            "top_areas": [{"area": a[0], "count": a[1]} for a in top_areas]
        }
    
    def compare(self, vacancy_ids: List[str]) -> Dict[str, Any]:
        """Сравнение вакансий"""
        vacancies = {v.get('Ids', ''): v for v in self.data}
        found = []
        
        for vid in vacancy_ids:
            if vid in vacancies:
                found.append(vacancies[vid])
        
        if len(found) < 2:
            return {"error": "Не найдено достаточно вакансий для сравнения"}
        
        comparison = {
            "count": len(found),
            "vacancies": []
        }
        
        for v in found:
            comparison["vacancies"].append({
                "id": v.get('Ids', ''),
                "name": v.get('Name', ''),
                "employer": v.get('Employer', ''),
                "salary_from": v.get('From', ''),
                "salary_to": v.get('To', ''),
                "experience": v.get('Experience', ''),
                "area": v.get('Area', ''),
                "schedule": v.get('Schedule', ''),
                "professional_roles": v.get('Professional roles', '')
            })
        
        return comparison


# Глобальный экземпляр для работы с данными
vacancy_data = VacancyData()


@mcp.tool()
def search_vacancies(
    query: str = Field(..., description="Поисковый запрос (название должности, ключевые слова)"),
    min_salary: Optional[float] = Field(None, description="Минимальная зарплата"),
    max_salary: Optional[float] = Field(None, description="Максимальная зарплата"),
    experience: Optional[str] = Field(None, description="Требуемый опыт работы"),
    area: Optional[str] = Field(None, description="Регион/город"),
    limit: int = Field(10, ge=1, le=100, description="Максимальное количество результатов")
) -> str:
    """
    Поиск вакансий по заданным критериям.
    
    Позволяет найти вакансии по названию, ключевым словам, зарплате, опыту и региону.
    Возвращает список подходящих вакансий с основной информацией.
    """
    try:
        params = SearchVacanciesParams(
            query=query,
            min_salary=min_salary,
            max_salary=max_salary,
            experience=experience,
            area=area,
            limit=limit
        )
        
        results = vacancy_data.search(params)
        
        if not results:
            return json.dumps({
                "status": "success",
                "message": "Вакансии не найдены",
                "count": 0,
                "results": []
            }, ensure_ascii=False, indent=2)
        
        formatted_results = []
        for v in results:
            formatted_results.append({
                "id": v.get('Ids', ''),
                "name": v.get('Name', ''),
                "employer": v.get('Employer', ''),
                "salary_from": v.get('From', ''),
                "salary_to": v.get('To', ''),
                "experience": v.get('Experience', ''),
                "area": v.get('Area', ''),
                "schedule": v.get('Schedule', ''),
                "professional_roles": v.get('Professional roles', ''),
                "description_preview": (v.get('Description', '') or '')[:200] + '...' if len(v.get('Description', '')) > 200 else v.get('Description', '')
            })
        
        return json.dumps({
            "status": "success",
            "count": len(formatted_results),
            "results": formatted_results
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Ошибка при поиске вакансий: {str(e)}"
        }, ensure_ascii=False)


@mcp.tool()
def get_vacancy_statistics(
    role: Optional[str] = Field(None, description="Профессиональная роль для фильтрации"),
    area: Optional[str] = Field(None, description="Регион для фильтрации"),
    specialization: Optional[str] = Field(None, description="Специализация для фильтрации")
) -> str:
    """
    Получение статистики по вакансиям.
    
    Возвращает общую статистику: количество вакансий, среднюю зарплату,
    диапазон зарплат, топ ролей и регионов.
    """
    try:
        params = GetVacancyStatsParams(
            role=role,
            area=area,
            specialization=specialization
        )
        
        stats = vacancy_data.get_stats(params)
        
        return json.dumps({
            "status": "success",
            "statistics": stats
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Ошибка при получении статистики: {str(e)}"
        }, ensure_ascii=False)


@mcp.tool()
def compare_vacancies(
    vacancy_ids: List[str] = Field(..., min_items=2, max_items=5, description="Список ID вакансий для сравнения")
) -> str:
    """
    Сравнение нескольких вакансий.
    
    Позволяет сравнить до 5 вакансий по основным параметрам:
    зарплата, опыт, регион, график работы и т.д.
    """
    try:
        params = CompareVacanciesParams(vacancy_ids=vacancy_ids)
        
        comparison = vacancy_data.compare(params.vacancy_ids)
        
        if "error" in comparison:
            return json.dumps({
                "status": "error",
                "message": comparison["error"]
            }, ensure_ascii=False)
        
        return json.dumps({
            "status": "success",
            "comparison": comparison
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Ошибка при сравнении вакансий: {str(e)}"
        }, ensure_ascii=False)


if __name__ == "__main__":
    # Запуск MCP-сервера
    mcp.run()

