
# chatbot reading existing memories

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from IPython.display import Image, display
from pathlib import Path

load_dotenv(override=True)

# step 1 - create LTM store + seeded memories (done before running the graph)
store = InMemoryStore()

user_id = "u1"

# store user's details as a single blob
# we can also split into multiple records; this keep it easy
user_details = ("user", user_id, "details")

store.put(user_details, "profile_1", {"data": "Name: Vikash"})
store.put(user_details, "profile_2", {"data": "Profession: AI Software Engineer"})
store.put(user_details, "preference_1", {"data": "Prefers concise answers"})
store.put(user_details, "preference_2", {"data": "Likes examples in Python"})
store.put(user_details, "project_1", {"data": "Building MCP servers (Python-based project)"})

# step 2 - system prompt template 
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user's preferences, context, and past interactions.

If the user's name or relevant personal context is available, always personalize your responses by:
    - Always Address the user by name (e.g., "Sure, Vikash...") when appropriate
    - Referencing known projects, tools, or preferences (e.g., "your MCP  server python based project")
    - Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..." 
say "Since your project is built with TypeScript..."

Use personalization especially in:
    - Greetings and transitions
    - Help or guidance tailored to tools and frameworks the user uses
    - Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

The user's memory (which may be empty) is provided as: {user_details_content}
"""

# step 3 - build graph : START --> chat --> END (read only LTM)
llm = ChatOpenAI(model="gpt-4o-mini")

# create node
def chat_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    # read-only: fetch user details memory (no writes)
    user_details = ("user", user_id, "details")
    items = store.search(user_details)

    # convert memory items into a string blob for {user_details_content}
    # keep it dead simple
    if items:
        user_details_content = "\n".join(f"- {it.value.get('data', '')}" for it in items)
    else:
        user_details_content = ""   # prompt says it may be empty

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        user_details_content=user_details_content
    )

    system_msg = SystemMessage(content=system_prompt)

    response = llm.invoke([system_msg] + state["messages"])

    return{
        "messages": [response]
    }

# build your graph workflow
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

# step 4 - run your graph workflow (provide user_id in config)
config = {
    "configurable": {"user_id": "u1"}
}

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Explain Gen AI in simple terms"}]},
    config,
)

print(f"result: {result["messages"][-1].content}")

# get your graph image
png_data = graph.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "ltm_chatbot_workflow.png", "wb") as f:
    f.write(png_data)

print(graph.get_graph().draw_mermaid())

