
# chatbot backend using LangGraph with Tool integration (HW)

import os
import requests
from typing import TypedDict, List, Annotated
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)
stock_price_api_key = os.getenv('STOCK_PRICE_API_KEY')
weather_api_key = os.getenv('WEATHER_API_KEY')

# ---------- State ----------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# ---------- Tools ----------
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_number: float, second_number: float, operation: str) -> float:
    """Perform basic arithmetic operations: add, subtract, multiply, divide."""
    if operation == "add":
        return first_number + second_number
    elif operation == "subtract":
        return first_number - second_number
    elif operation == "multiply":
        return first_number * second_number
    elif operation == "divide":
        return first_number / second_number
    else:
        raise ValueError("Invalid operation")

@tool
def get_stock_price(symbol: str) -> dict:
    """Get current stock price for a given company symbol (e.g., AAPL)."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_price_api_key}"
    return requests.get(url).json()

@tool
def get_weather(city: str) -> str:
    """Get current temperature in Celsius for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        return f"The current temperature in {city} is {temp}°C."
    else:
        return f"Could not fetch weather for {city}."

tools = [search_tool, calculator, get_stock_price, get_weather]

# ---------- LLM ----------
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# ---------- Nodes ----------
def chat_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ---------- Graph ----------
graph = StateGraph(AgentState)
graph.add_node("chat_node", chat_node)
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)
graph.add_edge("tools", "chat_node")

checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

# ---------- Optional: test ----------
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": "test"}}
    result = chatbot.invoke(
        {"messages": [HumanMessage(content="What is the weather in Buxar?")]},
        config=config
    )
    print(result["messages"][-1].content)