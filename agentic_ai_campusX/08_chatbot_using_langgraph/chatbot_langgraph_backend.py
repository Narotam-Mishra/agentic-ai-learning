
# chatbot backend using laggraph

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

# define your llm
llm = ChatOpenAI()

# define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']

    # send query to llm
    response = llm.invoke(messages)

    # store response to state
    return{
        'messages': [response]
    }

# checkpointer for memomry
checkpointer = InMemorySaver()

# create graph workflow
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile graph
simple_chatbot_workflow = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # get your graph image
    png_data = simple_chatbot_workflow.get_graph().draw_mermaid_png()
    display(Image(png_data))

    images_dir = Path(__file__).parent / "images"
    images_dir.mkdir(exist_ok=True)

    with open(images_dir / "simple_chatbot_workflow.png", "wb") as f:
        f.write(png_data)

    print(simple_chatbot_workflow.get_graph().draw_mermaid())

