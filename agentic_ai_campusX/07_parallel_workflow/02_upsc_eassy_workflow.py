
# upsc eassy parallel workflow

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from pydantic import BaseModel, Field
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

llm_model = ChatOpenAI(
    model='gpt-4o-mini'
)

class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the eassy')
    score: int = Field(description='Score out of 10', ge=0, le=10)

structured_model = llm_model.with_structured_output(EvaluationSchema)

# define graph
graph = StateGraph()

# compile the graph
eassy_workflow = graph.compile()

# get your graph image
png_data = eassy_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "eassy_workflow.png", "wb") as f:
    f.write(png_data)

print(eassy_workflow.get_graph().draw_mermaid())