
# short term memory implementation

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, MessagesState
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

checkpointer = InMemorySaver()

model = ChatOpenAI()

def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return{
        "messages": [response]
    }

builer = StateGraph(MessagesState)
builer.add_node("call_model", call_model)
builer.add_edge(START, "call_model")

graph = builer.compile(checkpointer=checkpointer)

config = {
    "configurable":{
        "thread_id": "thread-1"
    }
}

config2 = {
    "configurable":{
        "thread_id": "thread-2"
    }
}

res1 = graph.invoke({
    "messages": [{
        "role": "user",
        "content": "Hi! My name is Abhishek"
    }]},
    config
)

res2 = graph.invoke({
    "messages": [{
        "role": "user",
        "content": "What is my name?"
    }]},
    config2
)

# print(f"response: {res2}")

snap = graph.get_state(config2)
vals = snap.values
for m in vals.get("messages", []):
    print("-", type(m).__name__,":",m.content)

# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "stm_chatbot_workflow.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())
