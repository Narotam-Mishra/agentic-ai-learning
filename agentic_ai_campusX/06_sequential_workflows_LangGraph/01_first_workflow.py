
# BMI workflow

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display
from pathlib import Path

# define state
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str

# define utilities functions
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state['weight_kg']
    height = state['height_m']

    bmi = weight / (height**2)
    state['bmi'] = round(bmi, 2)
    return state

def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']

    if bmi < 18.5:
        state['category'] = "Underweight"
    elif 18.5 <= bmi < 25:
        state['category'] = "Normal"
    elif 25 <= bmi < 30:
        state['category'] = "Overweight"
    else:
        state['category'] = "Obese"

    return state


# define and register your graph
graph = StateGraph(BMIState)

# add nodes to your graph
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# add edges to your graph
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)

# compile the graph
graph_workflow = graph.compile()

# execute the graph
initial_state = {
    'weight_kg': 77,
    'height_m': 1.75,
}

final_state = graph_workflow.invoke(initial_state)

print(f"res: {final_state}")

# get your graph image
png_data = graph_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "bmi_workflow.png", "wb") as f:
    f.write(png_data)

print(graph_workflow.get_graph().draw_mermaid())
