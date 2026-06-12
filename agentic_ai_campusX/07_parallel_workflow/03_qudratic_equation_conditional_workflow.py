
# qudratic equation example using conditional workflow 

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
from IPython.display import Image, display
from pathlib import Path


load_dotenv(override=True)

class QuadState(TypedDict):
    a: int
    b: int
    c: int

    equation: str
    discriminant: float
    ans: str

def show_equation(state: QuadState):
    a = state["a"]
    b = state["b"]
    c = state["c"]

    equation = f"{a}x^2 {'+' if b >= 0 else '-'} {abs(b)}x {'+' if c >= 0 else '-'} {abs(c)} = 0"
    
    return{
        "equation": equation
    }

def calculate_discriminant(state: QuadState):
    discriminant = state["b"]**2 - (4*state["a"]*state["c"])
    
    return{
        "discriminant": discriminant
    }

def real_roots(state: QuadState):
    root1 = (-state["b"] + state["discriminant"]**0.5) / (2 * state["a"])
    root2 = (-state["b"] - state["discriminant"]**0.5) / (2 * state["a"])

    res = f'The roots are {root1} and {root2}'

    return{
        'ans': res
    }

def repeated_roots(state: QuadState):
    root = (-state["b"]) / (2 * state["a"])

    res = f'Only repeating root is {root}'

    return{
        'ans': res
    }


def no_real_roots(state: QuadState):
    res = f'No real roots'
    return{
        'ans': res
    }

def check_condition(state: QuadState) -> Literal["real_roots", "repeated_roots", "no_real_roots"]:
    if state['discriminant'] > 0:
        return "real_roots"
    elif state['discriminant'] == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"


# define graph
graph = StateGraph(QuadState)

# add nodes
graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

# add edges
graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')

# add conditional edges
graph.add_conditional_edges('calculate_discriminant', check_condition)
graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

quad_eq_conditional_workflow = graph.compile()

initial_state = {
    'a': 2,
    'b': 4,
    'c': 2
}

final_state = quad_eq_conditional_workflow.invoke(initial_state)
print(f"final_state: {final_state}")

# get your graph image
png_data = quad_eq_conditional_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "quad_eq_conditional_workflow.png", "wb") as f:
    f.write(png_data)

print(quad_eq_conditional_workflow.get_graph().draw_mermaid())
