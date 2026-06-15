
# persistence in LangGraph

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from typing import TypedDict
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

llm_model = ChatOpenAI()

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state: JokeState):
    prompt = f'generate a joke on the topic {state["topic"]}'
    response = llm_model.invoke(prompt).content

    return{
        'joke': response
    }

def generate_explanation(state: JokeState):
    prompt = f'write an explanation for the joke {state["joke"]}'
    response = llm_model.invoke(prompt).content

    return{
        'explanation': response
    }


# build graph
graph = StateGraph(JokeState)

# add nodes
graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

# add edges
graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

# intialize checkpointer
checkpointer = InMemorySaver()

# compile graph with checkpointer
persistence_workflow = graph.compile(checkpointer=checkpointer)

config1 = {
    'configurable': {"thread_id": "2"}
}

res = persistence_workflow.invoke(
    { 'topic': 'pasta' },
    config=config1
)

print(f"res: {res}")
# print(f"current state: {persistence_workflow.get_state(config1)}")
print(f"state history: {list(persistence_workflow.get_state_history(config1))}")

# get your graph image
png_data = persistence_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "persistence_workflow.png", "wb") as f:
    f.write(png_data)

print(persistence_workflow.get_graph().draw_mermaid())


