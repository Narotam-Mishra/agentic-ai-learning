
# persistence with crash in LangGraph

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from typing import TypedDict
from IPython.display import Image, display
from pathlib import Path
import time

load_dotenv(override=True)

llm_model = ChatOpenAI()

# step 1 - define state
class CrashState(TypedDict):
    input: str
    step1: str
    step2: str
    step3: str

# define steps
def step1(state: CrashState) -> CrashState:
    print(f"✅ step 1 executed...")
    return{
        "step1": "done",
        "input": state["input"]
    }

def step2(state: CrashState) -> CrashState:
    print(f"✅ step 2 hanging.... now manually interrupt from the IDE")
    
    # sleep for 30 seconds
    time.sleep(30)
    return{
        "step2": "done",
        "input": state["input"]
    }

def step3(state: CrashState) -> CrashState:
    print(f"✅ step 3 executed...")
    return{
        "step3": True,
    }

# build graph
graph_builder = StateGraph(CrashState)

# add nodes
graph_builder.add_node("step1", step1)
graph_builder.add_node("step2", step2)
graph_builder.add_node("step3", step3)

# add edges
graph_builder.add_edge(START, "step1")
graph_builder.add_edge("step1", "step2")
graph_builder.add_edge("step2", "step3")
graph_builder.add_edge("step3", END)

checkpointer = InMemorySaver()

# compile graph
crash_workflow = graph_builder.compile(checkpointer=checkpointer)

try:
    print(f"➡️Running graph: please manually interrupt during step 2...")
    crash_workflow.invoke(
        {'input': "start" },
        config={
            'configurable':{
                "thread_id": "thread-1"
            }
        }
    )
except KeyboardInterrupt:
    print(f"❌Kernel manually interrupted (creash simulated)")


# rerun the workflow (to show fault tolerance)
final_state = crash_workflow.invoke(None, config={
    "configurable": {"thread_id": "thread-1"}
})

print(f"final_state: {final_state}")

# print(f"current state: {crash_workflow.get_state({"configurable": {"thread_id": "thread-1"}})}")
# print(f"current history: {list(crash_workflow.get_state_history({"configurable": {"thread_id": "thread-1"}}))}")

# get your graph image
png_data = crash_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "crash_workflow.png", "wb") as f:
    f.write(png_data)

print(crash_workflow.get_graph().draw_mermaid())


