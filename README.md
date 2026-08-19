# AI Agent for Vacancy Analysis

AI agent for searching, analyzing, and comparing job vacancies through an MCP-based tool layer and an LLM API.

> Educational project developed for the Changellenge >> / Cloud.ru case **"MCP for Business AI Transformation"**.

## Overview

The project combines an LLM-powered agent with an MCP server that exposes structured vacancy operations. A user sends a request in natural language; the agent decides which tool to use, retrieves structured data, and generates the final response.

### Core capabilities

- Search vacancies by role, salary, experience, and location
- Calculate vacancy statistics
- Compare several vacancies
- Connect an LLM to external tools through MCP
- Return responses in an A2A-compatible structure
- Validate tool inputs with Pydantic
- Handle API and network errors

## Architecture

```text
User request
     |
     v
+----------------------+
|   HR AI Agent        |
|   Python + LLM API   |
+----------+-----------+
           |
           | MCP
           v
+----------------------+
|   Vacancy MCP Server |
|   Search / Stats /   |
|   Comparison tools   |
+----------+-----------+
           |
           v
     Vacancy dataset
```

### Main components

- `agent/hr_vacancies_agent.py` — asynchronous agent, LLM integration, tool calling, and A2A response formatting
- `mcp_server/vacancies_mcp.py` — MCP server exposing vacancy operations
- `test_agent.py` / `test_mcp.py` — basic project tests

## Tech Stack

**Language:** Python  
**AI / LLM:** Evolution Foundation Models API  
**Agent / Tooling:** MCP, FastMCP  
**Validation:** Pydantic  
**HTTP:** httpx  
**Configuration:** python-dotenv  
**Data:** CSV

## Installation

```bash
git clone https://github.com/Human221/resume-ml.git
cd resume-ml

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

Set the required API configuration in `.env` and provide a path to the vacancy dataset.

## Usage

Example request:

```python
import asyncio
from agent.hr_vacancies_agent import HRVacanciesAgent


async def main():
    agent = HRVacanciesAgent()
    result = await agent.process_query(
        "Найди вакансии Python разработчика с зарплатой от 100000 до 200000 рублей"
    )
    print(result)


asyncio.run(main())
```

Run the available tests:

```bash
python test_mcp.py
python test_agent.py
```

## Project Structure

```text
resume-ml/
├── agent/
│   └── hr_vacancies_agent.py
├── mcp_server/
│   └── vacancies_mcp.py
├── test_agent.py
├── test_mcp.py
├── tools.json
├── requirements.txt
├── requirements-train.txt
├── .env.example
└── README.md
```

Additional deployment and training notes are kept in the repository as separate guides.

## What I Worked On

The project demonstrates practical work with:

- LLM API integration
- Tool calling
- MCP server/client interaction
- Async Python
- Structured tool schemas
- Input validation
- Error handling
- Agent-to-tool orchestration

## Limitations

This repository is an educational case project rather than a production HR platform. The repository does not claim production-scale usage or measured business impact.

## License

Educational project created for the Changellenge >> / Cloud.ru case.
