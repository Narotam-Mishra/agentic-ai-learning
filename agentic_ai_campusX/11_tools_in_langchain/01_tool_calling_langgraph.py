
# Tool Calling in LangGraph

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random
import os
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)
stock_price_api_key = os.getenv('STOCK_PRICE_API_KEY')


# step 1 - LLM
llm_model = ChatOpenAI()

# step 2 - tools setup
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    perform a basic arithmetic operation on two numbers.
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
    fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_price_api_key}"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm_model.bind_tools(tools)

# define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# graph nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call"""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return{
        "messages": [response]
    }

# execute tool calls
tool_node = ToolNode(tools)

# graph node structure
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_node("tools", tool_node)

# add graph's edges
graph_builder.add_edge(START, "chat_node")

# add conditional edge for tool calling
graph_builder.add_conditional_edges("chat_node", tools_condition)

# create loop in the workflow
graph_builder.add_edge("tools", "chat_node")

chatbot_with_tool_workflow = graph_builder.compile()

# regular chat
res = chatbot_with_tool_workflow.invoke({
    "messages": [HumanMessage(content="What is stock price of Apple?")]
})

print(f"res: {res["messages"][-1].content}")

# get your graph image
png_data = chatbot_with_tool_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "chatbot_with_tool_workflow.png", "wb") as f:
    f.write(png_data)

print(chatbot_with_tool_workflow.get_graph().draw_mermaid())
