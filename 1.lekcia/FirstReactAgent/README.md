# Firstreactagent

A small Python project that demos calling OpenAI’s Chat Completions API with **function calling (tools)** to fetch weather data and convert temperatures.

## Features

* OpenAI Chat Completions with tool/function calling (`tools` + `tool_choice="auto"`).
* Example weather tools: `get_current_temperature`, `convert_to_celsius`, `get_city_forecast`.
* Environment configuration via `.env` using `python-dotenv`.
* Dependency management with **uv** and Python 3.12+.

---

## Requirements

* Python **3.12+**
* An OpenAI API key available as `OPENAI_API_KEY` (in `.env` for local dev)
* Dependencies (managed via uv): `openai`, `beautifulsoup4`, `python-dotenv`, `requests`

> The project declares these in `pyproject.toml`.

---

## Quick Start (with uv)

```bash
# 1) Create a virtual environment
uv venv

# 2) Install dependencies from pyproject
uv sync

# 3) Create your local env file
cp .env.example .env
# then edit .env and add your API key

# 4) Run your script
uv run main.py
```

---

## .env example

Create a `.env` file in the project root:

```dotenv
# .env
OPENAI_API_KEY=sk-your-real-key-here
```


# Weather Examples

## Example 1: Current temperature

```python
"What is the current temperature in Prague?"
```

* LLM calls:

  * `get_current_temperature("czech-republic/prague")`
  * `convert_to_celsius("<output>")`
* Return the current temperature in Prague.

---

## Example 2: Weather forecast

```python
"I'm planning to visit Prague next Friday. What will the weather be like?"
```

* LLM calls:

  * `get_city_forecast("czech-republic/prague")`
  * `convert_to_celsius("<output>")`
* Return the weather forecast for next Friday in Prague.

---

## Example 3: Multi-day forecast

```python
"Show me the weather forecast for Prague for the next 3 days."
```

* LLM calls:

  * `get_multi_day_forecast("czech-republic/prague", 3)`
  * `convert_to_celsius("<output>")`
* Return the weather forecast for Prague for the next 3 days.
