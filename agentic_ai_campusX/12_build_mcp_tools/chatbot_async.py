
# chatbot using langgraph with async capability

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import os
import asyncio

load_dotenv(override=True)
stock_price_api_key = os.getenv('STOCK_PRICE_API_KEY')
weather_api_key = os.getenv('WEATHER_API_KEY')

# step 1 - llm setup
llm = ChatOpenAI(model="gpt-5")

# step 2 - tool setup
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_price_api_key}"
    r = requests.get(url)
    return r.json()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f"https://api.weatherstack.com/current?access_key={weather_api_key}&query={city}"
  response = requests.get(url)

  return response.json()

tools = [search_tool, calculator, get_stock_price, get_weather_data]

# tool binding
llm_with_tools = llm.bind_tools(tools)

# step 3 - create and initialize state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def build_graph():
    
    # step 4 - create nodes
    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    # step 5 - graph setup
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")

    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge('tools', 'chat_node')

    # compile graph workflow
    chatbot = graph.compile()

    return chatbot

async def main():
    chatbot = build_graph()

    # running the graph
    res = await chatbot.ainvoke({
        "messages": [HumanMessage(content="Find the modulus of 132354 and 15 and give answer like a cricket commentator")]
    })

    print(f"res: {res['messages'][-1].content}")

if __name__ == '__main__':
    asyncio.run(main())


