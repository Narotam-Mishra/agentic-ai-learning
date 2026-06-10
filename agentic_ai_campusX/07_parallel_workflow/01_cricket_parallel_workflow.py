
from langgraph.graph import StateGraph
from typing import TypedDict

# create state
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    sr: float
    bpb: float
    boundary_percent: float

graph = StateGraph(BatsmanState)