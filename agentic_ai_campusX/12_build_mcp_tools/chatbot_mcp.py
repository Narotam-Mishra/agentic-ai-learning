
# chatbot using langgraph with mcp tool integration and async capability

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import os

load_dotenv(override=True)

expense_mcp_token = os.getenv("MCP_AUTH_TOKEN")
if not expense_mcp_token:
    raise RuntimeError(
        f"EXPENSE_MCP_TOKEN (or MCP_AUTH_TOKEN) is missing. "
        "Add the FastMCP deployment token before connecting to the hosted server."
    )

# step 1 - llm setup
llm = ChatOpenAI(model="gpt-5")

# step 2 - setup mcp client for local FastMCP server
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "/Users/narotamkumarmishra/Desktop/mycode-workspace/mcp-learning/basic-math-mcp-local-server/.venv/bin/python",
            "args": [
                "/Users/narotamkumarmishra/Desktop/mycode-workspace/mcp-learning/basic-math-mcp-local-server/main.py"
            ],
        },
        "manim-server": {
            "transport": "stdio",
            "command": "/Users/narotamkumarmishra/manim-env/bin/python",
            "args": [
                "/Users/narotamkumarmishra/Desktop/manim-mcp-server/src/manim_server.py"
            ],
            "env": {
                "MANIM_EXECUTABLE": "/Users/narotamkumashra/manim-env/bin/manim"
            },
        },
    }
)

# step 3 - create and initialize state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():

    tools = await client.get_tools()

    print(f"tools_list: {tools}")

    # tool binding
    llm_with_tools = llm.bind_tools(tools)
    
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
    chatbot = await build_graph()

    # running the graph
    res = await chatbot.ainvoke({
        "messages": [HumanMessage(content="Create a visual display for Bubble sorting using Manim tool")]
    })

    print(f"res: {res['messages'][-1].content}")

if __name__ == '__main__':
    asyncio.run(main())
