
# chatbot creating new memories

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

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[str] = Field(default_factory=list, description="Atomic user memories to store")

memory_extractor = extractor_llm.with_structured_output(MemoryDecision)

# step 3 - Graph: START -> remember -> END
# (Creates memories, but does NOT use them to answer)
def remember_only_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    namespace = ("user", user_id, "details")

    # take latest user's message
    lst_msg = state["messages"][-1].content

    # LLM decides what to store
    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(
                content=(
                    "Extract LONG-TERM memories from the user's message.\n"
                    "Only store stable, user-specific info (identity, preferences, ongoing projects).\n"
                    "Do NOT store transient info.\n"
                    "Return should_write=false if nothing is worth storing.\n"
                    "Each memory should be a short atomic sentence."
                )
            ),
            {"role": "user", "content": lst_msg}
        ]
    )

    # write to store (LTM)
    if decision.should_write:
        for mem in decision.memories:
            store.put(namespace, str(uuid.uuid4()), {"data": mem})

    # IMPORTANT: we are NOT using memory, not even responding with the LLM.
    # We just return a fixed acknowledgement.
    return{
        "messages": [{
            "role": "assistant", 
            "content": "Noted."
        }]
    }

# build graph workflow
builder = StateGraph(MessagesState)
builder.add_node("remember", remember_only_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", END)

graph = builder.compile(store=store)

# demonstration
config = {"configurable": {"user_id": "u1"}}

res1 = graph.invoke({"messages": [{"role": "user", "content": "Hi my name is Vikram"}]},config)
print("Assistant:", res1["messages"][-1].content)

res2 = graph.invoke({"messages": [{"role": "user", "content": "I am working as AI Software Engineer"}]},config)
print("Assistant:", res2["messages"][-1].content)

res3 = graph.invoke({"messages": [{"role": "user", "content": "My favorite programming language is Python"}]},config)
print("Assistant:", res3["messages"][-1].content)

items = store.search(("user", "u1", "details"))

for item in items:
    print(f"details: {item.value['data']}")

# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "ltm_chatbot_with_write_memory.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())




