
# blog writing agent (basic)

from __future__ import annotations
from typing import TypedDict, List, Annotated
import operator

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What is cover")


class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]