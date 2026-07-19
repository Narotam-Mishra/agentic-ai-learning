
# LTM with merged workflow for memory (with read and write operation)

from dotenv import load_dotenv
import uuid
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

# step 1 - create LTM store
store = InMemoryStore()

# step 2 - System Prompt
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user's preferences, context, and past interactions.

If the user's name or relevant personal context is available, always personalize your responses by:
    - Always Address the user by name (e.g., "Sure, Nitish...") when appropriate
    - Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    - Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible.

Use personalization especially in:
    - Greetings and transitions
    - Help or guidance tailored to tools and frameworks the user uses
    - Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

The user's memory (which may be empty) is provided as: {user_details_content}
"""

# step 3 - memory extraction from LLM
memory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if this memory is NEW and should be stored. False if duplicate/already known.")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")

memory_extractor = memory_llm.with_structured_output(MemoryDecision)

# add memory's system prompt
MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""

# step 4 - define nodes

# node 1 - remember
def remember_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # check for existing memory
    items = store.search(ns)
    existing = "\n".join(it.value["data"] for it in items) if items else "(empty)"

    # take latest user's message
    last_msg = state["messages"][-1].content

    # LLM decides what to store
    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(
                content=MEMORY_PROMPT.format(user_details_content=existing)),
                {"role": "user", "content": last_msg}
        ]
    )

    # write to store (LTM)
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(ns, str(uuid.uuid4()), {"data": mem.text})

    # return empty dict (no message).
    return{}

# node 2 - chat
chat_llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    user_details = "\n".join(it.value["data"] for it in items) if items else ""

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(
            user_details_content=user_details or "(empty)"
        )
    )

    response = chat_llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

# step 5 - define your graph workflow
builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

# step 6 - demonstration
config = {"configurable": {"user_id": "u1"}}

res1 = graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Vikash"}]}, config)
print(f"res1: {res1['messages'][-1].content}")

for it in store.search(("user", "u1", "details")):
    print(f"data1: {it.value["data"]}")

res2 = graph.invoke({"messages": [{"role": "user", "content": "I AI Software Engineer"}]}, config)
print(res2['messages'][-1].content)

for it in store.search(("user", "u1", "details")):
    print(f"data2: {it.value["data"]}")

res3 = graph.invoke({"messages": [{"role": "user", "content": "Explain GenAI simply"}]}, config)
print(res3['messages'][-1].content)

for it in store.search(("user", "u1", "details")):
    print(f"data3: {it.value["data"]}")

# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "ltm_chatbot_with_read_and_write_memory.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())

