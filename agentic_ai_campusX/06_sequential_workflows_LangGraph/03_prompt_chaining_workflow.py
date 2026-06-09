
# prompt chaining workflow

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

llm_model = ChatOpenAI()

class BlogState(TypedDict):
    title: str
    outline: str
    content: str

def create_outline(state: BlogState) -> BlogState:
    # fetch title
    title = state['title']

    # call llm and generate outline
    prompt = f"Generate a detailed outline for a blog on the topic - {title}"
    outline = llm_model.invoke(prompt)
    
    # update state
    state['outline'] = outline

    return state

def create_blog(state: BlogState) -> BlogState:
    # get title and outline from the state
    title = state['title']
    outline = state['outline']

    # create prompt
    prompt = f"Write a detailed blog on the title - {title} using the following outline \n {outline}"
    blog_content = llm_model.invoke(prompt).content

    # update state and return
    state['content'] = blog_content

    return state

# create graph
graph = StateGraph(BlogState)

# add nodes
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

# add edges
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

# compile graph
prompt_chain_workflow = graph.compile()

# execute graph
initial_state = {
    'title': 'Rise of AI in India'
}

final_state = prompt_chain_workflow.invoke(initial_state)
# print(f"final state: {final_state}")
# print(f"final outline: {final_state['outline']}")
print(f"final content: {final_state['content']}")

# get your graph image
png_data = prompt_chain_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "prompt_chain_workflow.png", "wb") as f:
    f.write(png_data)

print(prompt_chain_workflow.get_graph().draw_mermaid())