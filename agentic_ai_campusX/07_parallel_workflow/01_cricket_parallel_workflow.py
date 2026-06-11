
# parallel workflow example

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display
from pathlib import Path

# create state
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    sr: float
    bpb: float
    boundary_percent: float
    summary: str

def calculate_sr(state: BatsmanState) -> BatsmanState:
    sr = (state['runs'] / state['balls']) * 100
    return {'sr': sr}

def calculate_bpb(state: BatsmanState) -> BatsmanState:
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    return {'bpb': bpb}

def calculate_boundary_percent(state: BatsmanState) -> BatsmanState:
    boundary_percent = (((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs']) * 100
    return {'boundary_percent': boundary_percent}

def get_summary(state: BatsmanState) -> BatsmanState:
    summary = f"""
    Strike Rate - {state['sr']} \n
    Ball per boundary - {state['bpb']} \n
    Boundary percent - {state['boundary_percent']}
    """

    return {'summary': summary}

# define graph
graph = StateGraph(BatsmanState)

# add nodes
graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('get_summary', get_summary)

# add edges and build graph
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

graph.add_edge('calculate_sr', 'get_summary')
graph.add_edge('calculate_bpb', 'get_summary')
graph.add_edge('calculate_boundary_percent', 'get_summary')

graph.add_edge('get_summary', END)

# compile the graph
batsman_workflow = graph.compile()

# execute the graph
initial_state = {
    'runs': 109,
    'balls': 42,
    'fours': 7,
    'sixes': 9
}

final_state = batsman_workflow.invoke(initial_state)
print(f"res: {final_state}")

# get your graph image
png_data = batsman_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "batsman_workflow.png", "wb") as f:
    f.write(png_data)

print(batsman_workflow.get_graph().draw_mermaid())
