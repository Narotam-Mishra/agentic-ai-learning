
# chatbot workflow

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

# define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# define openAI LLM model
llm = ChatOpenAI()

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
checkpointer = MemorySaver()

# create graph workflow
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile graph
chatbot_workflow = graph.compile(checkpointer=checkpointer)

# initial_state = {
#     'messages': [
#         HumanMessage(content='What is the Capital of South Africa')
#     ]
# }

# set thread id 
thread_id = '9'

while True:
    user_message = input('type here... ')
    print(f"User's Message: {user_message}")

    if user_message.strip().lower() in ['exit', 'quit', 'bye']:
        break

    config = {
        'configurable':{
            'thread_id': thread_id,
        }
    }

    final_res = chatbot_workflow.invoke(
        {
            'messages': [HumanMessage(content=user_message)]
        },
        config=config
    )

    # print(f"AI Response: {final_res}")
    print(f"query_response: {final_res['messages'][-1].content}")
    # print(f"State: {chatbot_workflow.get_state(config=config)}")



# get your graph image
png_data = chatbot_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "chatbot_workflow.png", "wb") as f:
    f.write(png_data)

print(chatbot_workflow.get_graph().draw_mermaid())


