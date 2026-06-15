
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
time_travel_workflow = graph.compile(checkpointer=checkpointer)

config1 = {
    'configurable': {"thread_id": "2"}
}

res = time_travel_workflow.invoke(
    { 'topic': 'pasta' },
    config=config1
)

print(f"res: {res}")
print(f"current state: {time_travel_workflow.get_state(config1)}")

state_history = list(time_travel_workflow.get_state_history(config1))

# time travel using checkpointer_id
print("\nAvailable checkpoints:")
for index, snapshot in enumerate(state_history):
    checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
    print(f"{index}: {checkpoint_id}, next: {snapshot.next}")

# Pick the checkpoint where the next node is generate_joke.
# Updating this checkpoint lets us change the topic before joke generation.
selected_snapshot = next(
    snapshot for snapshot in state_history
    if "generate_joke" in snapshot.next
)
selected_checkpoint_id = selected_snapshot.config["configurable"]["checkpoint_id"]
selected_config = selected_snapshot.config

state_at_checkpoint = time_travel_workflow.get_state(selected_config)

print(f"\nSelected checkpoint id: {selected_checkpoint_id}")
print(f"state at selected checkpoint: {state_at_checkpoint}")

print("*****************************************************************")
# reexecute from selected checkpoint_id
reexecuted_result = time_travel_workflow.invoke(None, selected_config)
print(f"reexecuted result: {reexecuted_result}")

# update state using checkpointer id
print(f"{'*' * 50}")
updated_topic = "paneer"

updated_config = time_travel_workflow.update_state(
    selected_config,
    {"topic": updated_topic},
)

updated_state = time_travel_workflow.get_state(updated_config)
print(f"updated checkpoint config: {updated_config}")
print(f"updated state: {updated_state}")

reexecuted_updated_result = time_travel_workflow.invoke(None, updated_config)
print(f"reexecuted updated result: {reexecuted_updated_result}")
print(f"joke for updated topic '{updated_topic}': {reexecuted_updated_result['joke']}")

# state history
print(f"{'*' * 40}")
print(f"state history: {state_history}")

# get your graph image
png_data = time_travel_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "time_travel_workflow.png", "wb") as f:
    f.write(png_data)

print(time_travel_workflow.get_graph().draw_mermaid())
