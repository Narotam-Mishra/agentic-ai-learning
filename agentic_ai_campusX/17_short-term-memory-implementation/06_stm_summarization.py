
# short term memory with summaization and deletion

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, RemoveMessage
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

model = ChatOpenAI()

class ChatState(MessagesState):
    summary: str

def chat_node(state: ChatState):
    messages = []
    summary = state.get("summary", "")

    if summary:
        messages.append({
            "role": "system",
            "content": f"Conversation summary: \n{summary}"
        })

    messages.extend(state["messages"])
    print(f"Messgaes: {messages}")

    response = model.invoke(messages)
    return{
        "messages": [response]
    }

def summarize_conversation(state: ChatState):
    existing_summary = state.get("summary", "")

    # build summarization prompt
    if existing_summary:
        prompt = (
            f"Existing summary: \n{existing_summary}\n\n"
            "Extend the summary using the new conversation above."
        )
    else:
        prompt = "Summarize the conversation above"

    messages_for_summary = state["messages"] + [
        HumanMessage(content=prompt)
    ]

    response = model.invoke(messages_for_summary)

    # keep only last 2 messgaes verbatim
    messages_to_delete = state["messages"][:-2]

    return{
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete],
    }

def should_summarize(state: ChatState):
    return len(state["messages"]) > 6


builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "chat")

builder.add_conditional_edges(
    "chat",
    should_summarize,
    {
        True: "summarize",
        False: "__end__",
    }
)

builder.add_edge("summarize", "__end__")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "t1"}}

# gives the current version of the state
def show_state():
    snap = graph.get_state(config)
    vals = snap.values
    print("\n-- STATE ---")
    print("summary:", vals.get("summary", ""))
    print("num_messages:", len(vals.get("messages", [])))
    print("messages:")
    for m in vals.get("messages", []):
        print("-", type(m).__name__, ":", m.content[:80])

out1 = graph.invoke(
    {
        "messages": [HumanMessage(content="Quantum Physics")],
        "summary": ''
    },
    config=config
)

print(f"output: {out1}")
show_state()

out2 = graph.invoke(
    {
        "messages": [HumanMessage(content="How is Albert Einstein related?")],
    },
    config=config
)

print(f"output: {out2}")
show_state()

out3 = graph.invoke(
    {
        "messages": [HumanMessage(content="What are some of Einstein's famous work?")],
    },
    config=config
)

print(f"output: {out3}")
show_state()

out4 = graph.invoke(
    {
        "messages": [HumanMessage(content="Explain special theory of relativity?")],
    },
    config=config
)

print(f"output: {out4}")
show_state()


# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "stm_with_summarization.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())
