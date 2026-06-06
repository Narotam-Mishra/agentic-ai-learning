import os
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


# Step 1: Load environment variables from the local .env file.
# Required: OPENAI_API_KEY
# Optional: OPENAI_MODEL, for example gpt-4o-mini
load_dotenv(Path(__file__).with_name(".env"), override=True)


# Step 2: Define the state shared by all LangGraph nodes.
class JDState(TypedDict, total=False):
    prompt: str
    jd: str
    approved: bool


# Step 3: Create the LLM used by the graph.
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)


# Step 4: First node - receive or create the hiring request.
def hiring_request(state: JDState) -> JDState:
    return {
        **state,
        "prompt": "We need to hire a software engineer for the backend team.",
    }


# Step 5: Second node - generate a job description using the LLM.
def create_jd(state: JDState) -> JDState:
    prompt = state["prompt"]
    response = llm.invoke(f"Create a job description for this: {prompt}")

    return {
        **state,
        "jd": response.content,
    }


# Step 6: Third node - check whether the JD is approved.
# This uses dummy logic for learning: approve if the JD contains "engineer".
def check_approval(state: JDState) -> JDState:
    jd_text = state["jd"]
    approved = "engineer" in jd_text.lower()

    return {
        **state,
        "approved": approved,
    }


# Step 7: Router - choose the next node based on approval status.
def approval_router(state: JDState) -> Literal["approved", "not_approved"]:
    return "approved" if state["approved"] else "not_approved"


# Step 8: Final node - post or print the approved JD.
def post_jd(state: JDState) -> JDState:
    print("\nFinal Approved JD:\n")
    print(state["jd"])
    return state


# Step 9: Build the LangGraph workflow.
graph_builder = StateGraph(JDState)

graph_builder.add_node("HiringRequest", hiring_request)
graph_builder.add_node("CreateJD", create_jd)
graph_builder.add_node("CheckApproval", check_approval)
graph_builder.add_node("PostJD", post_jd)

graph_builder.add_edge(START, "HiringRequest")
graph_builder.add_edge("HiringRequest", "CreateJD")
graph_builder.add_edge("CreateJD", "CheckApproval")

graph_builder.add_conditional_edges(
    "CheckApproval",
    approval_router,
    {
        "approved": "PostJD",
        "not_approved": "CreateJD",
    },
)

graph_builder.add_edge("PostJD", END)


# Step 10: Compile the graph into a runnable app.
app = graph_builder.compile()


def main() -> None:
    # Step 11: Validate configuration before calling the LLM.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your environment or a .env file."
        )

    # Step 12: Run the graph.
    final_state = app.invoke({})

    print("\nWorkflow Completed.")
    print(f"Approved: {final_state['approved']}")


# Step 13: Start only when this file is run directly.
if __name__ == "__main__":
    main()
