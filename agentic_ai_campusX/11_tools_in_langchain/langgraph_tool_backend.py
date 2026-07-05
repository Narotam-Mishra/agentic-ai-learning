
# chatbot backend using LangGraph with Tool integration

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from IPython.display import Image, display
from pathlib import Path
import sqlite3
import requests
import os


load_dotenv(override=True)
stock_price_api_key = os.getenv('STOCK_PRICE_API_KEY')
weather_api_key = os.getenv('WEATHER_API_KEY')

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------
# 2. Tools
# -------------------
# Tools
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


@tool
def list_github_prs(
    owner: str,
    repo: str,
    state: str = "open",
    per_page: int = 5,
) -> list[dict]:
    """List the latest pull requests for a GitHub repository.

    Args:
        owner: GitHub organization or username (for example, ``langgraph-ai``).
        repo: Repository name (for example, ``langgraph``).
        state: Pull request state: ``open``, ``closed``, or ``all``.
        per_page: Number of pull requests to fetch, from 1 to 100.

    Returns:
        A simplified list containing each pull request's number, title,
        author, state, and URL.
    """
    if state not in {"open", "closed", "all"}:
        raise ValueError("state must be 'open', 'closed', or 'all'")
    if not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    response = requests.get(
        url,
        headers=headers,
        params={"state": state, "per_page": per_page, "sort": "created", "direction": "desc"},
        timeout=10,
    )
    response.raise_for_status()

    return [
        {
            "number": pr["number"],
            "title": pr["title"],
            "author": pr["user"]["login"],
            "state": pr["state"],
            "url": pr["html_url"],
        }
        for pr in response.json()
    ]


# tools setup
tools = [search_tool, calculator, get_stock_price, get_weather_data]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot_workflow_with_tool = graph.compile(checkpointer=checkpointer)
simple_chatbot_workflow = chatbot_workflow_with_tool

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

if __name__ == "__main__":
    # get your graph image
    png_data = chatbot_workflow_with_tool.get_graph().draw_mermaid_png()
    display(Image(png_data))

    images_dir = Path(__file__).parent / "images"
    images_dir.mkdir(exist_ok=True)

    with open(images_dir / "chatbot_workflow_with_tool.png", "wb") as f:
        f.write(png_data)

    print(chatbot_workflow_with_tool.get_graph().draw_mermaid())
