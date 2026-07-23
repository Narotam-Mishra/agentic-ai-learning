
# blog writing agent (basic)

from __future__ import annotations
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
import operator

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What is cover")


class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    # reducer: results from workers get concatenated automatically
    sections: Annotated[List[str], operator.add]
    final: str

llm = ChatOpenAI(model="gpt-5.1")

def orchestrator(state: State) -> dict:
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "Create a blog plan with 5-7 sections on the following topic."
                )
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )
    return{
        "plan": plan
    }

def fanout(state: State):
    return[
        Send("worker", {
            "task": task,
            "topic": state["topic"],
            "plan": state["plan"]
        })
        for task in state["plan"].tasks
    ]

def worker(payload: dict) -> dict:

    # payload contains what we sent
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = llm.invoke(
        [
            SystemMessage(content="Write one clean Markdown section"),
            HumanMessage(
                content=(
                    f"Blog: {blog_title}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n\n"
                    "Return only the section content in Markdown."
                )
            ),
        ]
    ).content.strip()

    return{
        "sections": [section_md]
    }

def reducer(state: State) -> dict:
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    # save to file
    filename = title.lower().replace(" ", "-") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return{
        "final": final_md
    }

# define your langgraph workflow

# add nodes
g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

# add edges
g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

out = app.invoke({
    "topic": "Write a blog on Self Attention",
    "sections": []
})

print(f"result: {out}")

# get your graph image
png_data = app.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "blog-writing-agent.png", "wb") as f:
    f.write(png_data)


