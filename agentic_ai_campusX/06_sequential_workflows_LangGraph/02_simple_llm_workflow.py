
# simple LLM workflow

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

llm_model = ChatOpenAI()

# create a state
class LLMState(TypedDict):
    query: str
    answer: str

def llm_qa(state: LLMState) -> LLMState:
    # extract the question from state
    question = state['query']

    # form a prompt
    prompt = f"Answer the following question {question}"

    # ask that question to the LLM
    answer = llm_model.invoke(prompt).content

    # update the answer in the state
    state['answer'] = answer

    return state

# create graph
graph = StateGraph(LLMState)

# add nodes
graph.add_node('llm_qa', llm_qa)

# add edges
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

# compile graph
llm_workflow = graph.compile()

# execute graph
initial_state = {
    "query": 'How far is the Moon from the Earth?'
}

final_res = llm_workflow.invoke(initial_state)
print(f"res: {final_res['answer']}")

# get your graph image
png_data = llm_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "llm_workflow.png", "wb") as f:
    f.write(png_data)

print(llm_workflow.get_graph().draw_mermaid())