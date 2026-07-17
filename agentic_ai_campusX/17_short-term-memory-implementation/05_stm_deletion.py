
# short term memory with deletion

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

model = ChatOpenAI()

def chat(state: MessagesState):
    response = model.invoke(state["messages"])
    return{
        "messages": [response]
    }

def delete_old_messages(state: MessagesState):
    msgs = state["messages"]

    # if more than 10 messages, then delete the earliest 6
    if len(msgs) > 10:
        to_remove = msgs[:6]
        return{
            "messages": [RemoveMessage(id=m.id) for m in to_remove]
        }
    
    return{}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_node("cleanup", delete_old_messages)

builder.add_edge(START, "chat")
builder.add_edge("chat", "cleanup") # run deletion after each response
builder.add_edge("cleanup", "__end__")

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "t1"}}

# run multiple turns
graph.invoke({"messages": [{"role": "user", "content": "Hi, I'm Vikash"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "Tell me about LangGraph"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "Now explain checkpointers"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is Langchain"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is Quantum Mechanics"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is Gen AI"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is my name"}]}, config)

snap = graph.get_state(config)
print("stored messages after cleanup: ", len(snap.values["messages"]))

# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "stm_with_deletion.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())
