
# Human In Loop implementation

from dotenv import load_dotenv
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4.1-mini")

# step 1 - define chat state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# step 2 - define chat node
def chat_node(state: ChatState):
    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a user question",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? yes/no"
    })

    if decision["approved"] == "no":
        return {
            "messages": [AIMessage(content="Not approved.")]
        }
    else:
        response = llm.invoke(state["messages"])
        return{
            "messages": [response]
        }
    
# step 3 - build the graph: START -> chat -> END
builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)

builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# Checkpointer is required for interrupts
checkpointer = MemorySaver()

# compile workflow
app = builder.compile(checkpointer=checkpointer)

# create a new thread id for this conversation
config = {
    "configurable": {"thread_id": '1234'}
}

# user asks a question
initial_input = {
    "messages": [
        ("user", "Explain gradient descent in simple terms")
    ]
}

# invoke the graph for the first time
result = app.invoke(initial_input, config=config)
# print(f"result: {result}")

# extract interrrupt messgae
message = result['__interrupt__'][0].value
# print(f"interrupt_message: {message}")

user_input = input(f"\nBackend Message - {message} \n Approve this question? (y/n): ")

# resume the graph with the approval decision
final_response = app.invoke(
    Command(resume={
        "approved": user_input
    }),
    config=config
)

print(f"final_result: {final_response}")

# get your graph image
png_data = app.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "hitl_chatbot_workflow.png", "wb") as f:
    f.write(png_data)

print(app.get_graph().draw_mermaid())
