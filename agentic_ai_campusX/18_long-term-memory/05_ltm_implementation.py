
# chatbot creating new memories without duplication

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

# step 2 - LLM that decides what to remember (structured output)
extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if this memory is NEW and should be stored. False if duplicate/already known.")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")

memory_extractor = extractor_llm.with_structured_output(MemoryDecision)

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
- If there is nothing memory-worthy, return an empty list.
"""

def chat_creates_memory_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # extract user id
    user_id = config["configurable"]["user_id"]

    namespace = ("user", user_id, "details")

    # a. load existing memories
    existing_items = store.search(namespace)
    existing_texts = [it.value.get("data", "") for it in existing_items if it.value.get("data")]
    user_details_content = "\n".join(f"- {t}" for t in existing_texts) if existing_texts else "(empty)"

    # b. latest user's message
    last_text = state["messages"][-1]

    # c. LLM extracts memories + marks new vs duplicate
    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=user_details_content)),
            {"role": "user", "content": f"USER MESSAGE:\n{last_text}"},
        ]
    )

    # d. store only new memory
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(namespace, str(uuid.uuid4()), {"data": mem.text})

    return{
        "messages": [{"role": "assistant", "content": "Noted."}]
    }

# Build graph: START -> chat -> END
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_creates_memory_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

config = {"configurable": {"user_id": "u1"}}

r1 = graph.invoke({"messages": [{"role": "user", "content": "My name is Vikash"}]}, config)
print("Assistant:", r1["messages"][-1].content)

# Likely duplicate of the first memory -> is_new should be false -> saved=0
r2 = graph.invoke({"messages": [{"role": "user", "content": "I like Python for programming."}]}, config)
print("\nAssistant:", r2["messages"][-1].content)

for it in store.search(("user", "u1", "details")):
    print(f"data: {it.value['data']}")

# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "ltm_chatbot_with_write_memory_without_duplicates.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())