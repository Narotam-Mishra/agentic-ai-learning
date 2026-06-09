
# prompt chaining workflow (homework)

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)
model = ChatOpenAI(model="gpt-4")

class BlogState(TypedDict):
    topic: str
    outline: str
    content: str
    score: int

def create_outline(state: BlogState) -> BlogState:
    topic = state["topic"]
    prompt = f"Generate a detailed outline for a blog on: {topic}"
    response = model.invoke(prompt)
    state["outline"] = response.content
    return state

def create_blog(state: BlogState) -> BlogState:
    topic = state["topic"]
    outline = state["outline"]
    prompt = f"Write a detailed blog on '{topic}' using this outline:\n{outline}"
    response = model.invoke(prompt)
    state["content"] = response.content
    return state

def evaluate_blog(state: BlogState) -> BlogState:
    outline = state["outline"]
    content = state["content"]
    prompt = f"""
    Rate the blog (1-10) based on how well it follows the outline and quality.
    Outline: {outline}
    Blog: {content}
    Output only the integer score.
    """
    response = model.invoke(prompt)
    try:
        state["score"] = int(response.content.strip())
    except:
        state["score"] = 5
    return state

# Build graph
graph = StateGraph(BlogState)
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)
graph.add_node("evaluate_blog", evaluate_blog)

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", "evaluate_blog")
graph.add_edge("evaluate_blog", END)

workflow_hw = graph.compile()

# Run
initial = {"topic": "Rise of AI in India", "outline": "", "content": "", "score": 0}
final = workflow_hw.invoke(initial)
print(f"Score: {final['score']}/10")

# get your graph image
png_data = workflow_hw.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "prompt_chain_workflow_hw.png", "wb") as f:
    f.write(png_data)

print(workflow_hw.get_graph().draw_mermaid())