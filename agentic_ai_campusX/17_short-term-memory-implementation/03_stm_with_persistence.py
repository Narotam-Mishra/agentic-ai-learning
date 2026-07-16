
# short term memory with persistence

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
import os

load_dotenv(override=True)

DB_URI = os.getenv("DB_URI")

llm = ChatOpenAI(model="gpt-4o-mini")

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return{
        "messages": [response]
    }

# build graph
builer = StateGraph(MessagesState)
builer.add_node("call_model", call_model)
builer.add_edge(START, "call_model")

t2 = {"configurable": {"thread_id": "thread-2"}}

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Create or migrate the LangGraph checkpoint tables before reading state.
    checkpointer.setup()

    graph = builer.compile(checkpointer=checkpointer)

    out2 = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, t2)
    # print("Thread-2:", out2["messages"][-1].content)


    snap = graph.get_state(t2)
    msg = snap.values.get("messages", [])
    print("Last message:", msg[-1].content if msg else None)

