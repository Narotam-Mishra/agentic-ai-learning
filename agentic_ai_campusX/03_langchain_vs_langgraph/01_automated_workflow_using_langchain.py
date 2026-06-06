
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# Step 1: Load environment variables from .env file.
# This lets ChatOpenAI read OPENAI_API_KEY and optional OPENAI_MODEL.
load_dotenv(override=True)


# Step 2: Define the approval function.
# This simulates a human-in-the-loop approval step.
def approve_jd(jd: str) -> bool:
    """
    Simulate approval decision.
    For demo, we ask the user. You can replace with automatic rules.
    """
    print("\n--- Review the JD above ---")
    user_input = input("Approve this JD? (y/n): ").strip().lower()
    return user_input == 'y'


# Step 3: Define the final action after approval.
# In a real app, this could call LinkedIn, Naukri, Workday, etc.
def post_jd(jd: str) -> None:
    """Simulate posting the JD to job portals."""
    print("\n✅ JD Approved and Posted to job portals!\n")
    # In a real scenario, you would call LinkedIn API, etc.


# Step 4: Build the LangChain pipeline.
# Flow: prompt template -> OpenAI chat model -> string output parser.
def build_jd_chain():
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    jd_prompt = ChatPromptTemplate.from_template(
        "Create a job description based on the hiring request:\n\n{request}"
    )
    return jd_prompt | llm | StrOutputParser()


def main() -> None:
    # Step 5: Validate required configuration before calling the model.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your environment or a .env file."
        )

    # Step 6: Define the hiring request and create the JD generation chain.
    hiring_prompt = "We need to hire a Software Backend Engineer for our backend team."
    jd_chain = build_jd_chain()

    # Step 7: Keep generating a JD until the user approves it.
    approved = False
    jd_output = None

    while not approved:
        # Step 8: Invoke the chain with the hiring request.
        jd_output = jd_chain.invoke({"request": hiring_prompt})

        # Step 9: Show the generated JD to the user for review.
        print("\n" + "="*50)
        print("GENERATED JOB DESCRIPTION:\n")
        print(jd_output)
        print("="*50)

        # Step 10: Ask the user whether to approve or regenerate.
        approved = approve_jd(jd_output)
        if not approved:
            print("\n🔄 JD not approved. Regenerating...\n")

    # Step 11: Post the final approved JD.
    post_jd(jd_output)


# Step 12: Start the script only when this file is run directly.
if __name__ == "__main__":
    main()
