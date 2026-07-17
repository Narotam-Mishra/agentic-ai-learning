
# solve context overflow problem using trimming

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

model = ChatOpenAI()

MAX_TOKENS = 151

def call_model(state: MessagesState):
    # trim conversation history: last N messages that fit within the token budget
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS
    )

    print('Current Token Count -->', count_tokens_approximately(messages=messages))

    for message in messages:
        print(f"Message Content: {message.content}")

    response = model.invoke(messages)
    return{
        "messages": [response]
    }

# build graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable":{
        "thread_id": "chat-1"
    }
}

result1 = graph.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Hi, My name is Vikash"
        }]
    },
    config
)

result2 = graph.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "I am learning LangGraph"
        }]
    },
    config
)

result3 = graph.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Can You explain short term memory?"
        }]
    },
    config
)

result4 = graph.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Explain MCP in details?"
        }]
    },
    config
)

print(f"result: {result3["messages"][-1].content}")

# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "stm_chatbot_with_trimming.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())
