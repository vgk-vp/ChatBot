"""Agent tools for search, calculation, and summarization."""

import json
import numexpr as ne
from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS
import os

class Config:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

config = Config()

# Initialize LLM for summarization
llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0.1)

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results with title, url, snippet."""
    items = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                items.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        return f"Search error: {e}"
    
    if not items:
        return "No search results found."
    
    return json.dumps(items, ensure_ascii=False, indent=2)

@tool
def calculate(expression: str) -> str:
    """Safely evaluate arithmetic expressions: + - * / % ^ and parentheses."""
    # Clean the expression
    expression = expression.strip()
    
    # Allow only safe characters
    allowed = set("0123456789.+-*/()%^ ")
    if any(ch not in allowed for ch in expression):
        return "Only arithmetic expressions with + - * / % ^ and parentheses are allowed."
    
    try:
        # Use numexpr for safe evaluation
        val = ne.evaluate(expression)
        result = val.item() if hasattr(val, "item") else val
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"

@tool
def summarize_text(text: str) -> str:
    """Summarize the given text into 3-5 concise, factual bullet points."""
    if len(text.strip()) < 50:
        return "Text too short to summarize meaningfully."
    
    # Limit text length to avoid token issues
    text = text[:3000] if len(text) > 3000 else text
    
    prompt = (
        "Summarize the following text into 3-5 concise, factual bullet points. "
        "Focus on the key information and main points. Be direct and factual.\n\n"
        f"TEXT TO SUMMARIZE:\n{text}"
    )
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content
    except Exception as e:
        return f"Error summarizing text: {e}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def weather_info(location: str) -> str:
    """Get weather information for a location using web search."""
    query = f"weather {location} today current conditions"
    return web_search(query)

@tool
def news_search(topic: str) -> str:
    """Search for recent news about a specific topic."""
    query = f"latest news {topic} 2024"
    return web_search(query)

@tool
def definition_lookup(term: str) -> str:
    """Look up the definition of a term or concept."""
    query = f"define {term} meaning definition"
    return web_search(query)

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units (length, weight, temperature)."""
    # Simple unit conversion logic
    conversions = {
        # Length (to meters)
        "mm": 0.001, "cm": 0.01, "m": 1, "km": 1000,
        "inch": 0.0254, "ft": 0.3048, "yard": 0.9144, "mile": 1609.34,
        
        # Weight (to grams)
        "mg": 0.001, "g": 1, "kg": 1000,
        "oz": 28.3495, "lb": 453.592,
        
        # Temperature handled separately
    }
    
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # Handle temperature conversions
    if from_unit in ["celsius", "c"] and to_unit in ["fahrenheit", "f"]:
        result = (value * 9/5) + 32
        return f"{value}°C = {result:.2f}°F"
    elif from_unit in ["fahrenheit", "f"] and to_unit in ["celsius", "c"]:
        result = (value - 32) * 5/9
        return f"{value}°F = {result:.2f}°C"
    
    # Handle other unit conversions
    if from_unit in conversions and to_unit in conversions:
        # Convert to base unit, then to target unit
        base_value = value * conversions[from_unit]
        result = base_value / conversions[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    
    return f"Conversion from {from_unit} to {to_unit} not supported. Supported units: length (mm, cm, m, km, inch, ft, yard, mile), weight (mg, g, kg, oz, lb), temperature (celsius, fahrenheit)"

# Tool registry for easy access
TOOLS = {
    "web_search": web_search,
    "calculate": calculate,
    "summarize_text": summarize_text,
    "get_current_time": get_current_time,
    "weather_info": weather_info,
    "news_search": news_search,
    "definition_lookup": definition_lookup,
    "unit_converter": unit_converter,
}

def get_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all available tools."""
    return {
        "web_search": "Search the web for current information and facts",
        "calculate": "Perform mathematical calculations and arithmetic",
        "summarize_text": "Create bullet point summaries of long text",
        "get_current_time": "Get the current date and time",
        "weather_info": "Get weather information for any location",
        "news_search": "Search for recent news about specific topics",
        "definition_lookup": "Look up definitions and meanings of terms",
        "unit_converter": "Convert between different units (length, weight, temperature)"
    }