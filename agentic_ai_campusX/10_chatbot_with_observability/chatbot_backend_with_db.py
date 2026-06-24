
# chatbot backend using laggraph

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from IPython.display import Image, display
from pathlib import Path
import sqlite3

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

# connect to sqlite db
connection_obj = sqlite3.connect('langgraph_chatbot_db', check_same_thread=False)

# checkpointer for db
checkpointer = SqliteSaver(conn=connection_obj)

# create graph workflow
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile graph
simple_chatbot_workflow = graph.compile(checkpointer=checkpointer)

# define config
CONFIG = {
    'configurable':{
        'thread_id': 'thread-07'
    }
}

# test db setup
response = simple_chatbot_workflow.invoke(
    {'messages': [HumanMessage(content='What is my name?')]},
    config = CONFIG,
)

# print(f"res: {response}")
# print(f"threads: {checkpointer.list(None)}")

# retrieve all unique threads in list
def retrieve_all_threads():
    all_unique_threads = set()
    for checkpoint in checkpointer.list(None):
        all_unique_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_unique_threads) 


if __name__ == "__main__":
    # get your graph image
    png_data = simple_chatbot_workflow.get_graph().draw_mermaid_png()
    display(Image(png_data))

    images_dir = Path(__file__).parent / "images"
    images_dir.mkdir(exist_ok=True)

    with open(images_dir / "langgraph_chatbot_workflow.png", "wb") as f:
        f.write(png_data)

    print(simple_chatbot_workflow.get_graph().draw_mermaid())

