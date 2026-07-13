
# subgraph implementation in LangGraph (with shared state)

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str

parent_llm = ChatOpenAI(model='gpt-4o-mini')
subgraph_llm = ChatOpenAI(model='gpt-4o')

def translate_text(state: ParentState):
    prompt = f"""
        Translate the following text to Hindi.
        Keep it natural and clear. Do not add extra content.

        Text:
        {state["answer_eng"]}
        """.strip()
    
    translated_text = subgraph_llm.invoke(prompt).content
   
    return {
        'answer_hin': translated_text
    }

subgraph_builder = StateGraph(ParentState)

subgraph_builder.add_node('translate_text', translate_text)

subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)

subgraph = subgraph_builder.compile()

def generate_answer(state: ParentState):
    answer = parent_llm.invoke(f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}").content
    return {
        'answer_eng': answer
    }

parent_builder = StateGraph(ParentState)

parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", subgraph)

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)

graph = parent_builder.compile()

res = graph.invoke({'question': 'What is quantum computing?'})
print(f"result: {res}")

# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "subgraph_shared_workflow.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())