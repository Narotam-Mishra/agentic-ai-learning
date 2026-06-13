
# post generator using iterative workflow

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from IPython.display import Image, display
from pathlib import Path
import operator

load_dotenv(override=True) 

# three LLMs instance for each workflow
generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
evaluator_llm = ChatOpenAI(model="gpt-4.1")
optimizer_llm = ChatOpenAI(model="gpt-4.1")

# create tweet evaluation schema
class TweetEvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the tweet.")

# using structured output
structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluationSchema)

# define state
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

def generate_tweet(state: TweetState):
    # prompt to create tweet
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
        Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

        Rules:
            - Do NOT use question-answer format.
            - Max 280 characters.
            - Use observational humor, irony, sarcasm, or cultural references.
            - Think in meme logic, punchlines, or relatable takes.
            - Use simple, day to day english
        """)
    ]


    # send prompt to generator llm
    response = generator_llm.invoke(messages).content

    # return response
    return{
        'tweet': response,
        'tweet_history': [response]
    }

def evaluate_tweet(state: TweetState):
    # prompt to evaluate tweet
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""Evaluate the following tweet: Tweet: "{state['tweet']}"
        Use the criteria below to evaluate the tweet:
            1. Originality - Is this fresh, or have you seen it a hundred times before?  
            2. Humor - Did it genuinely make you smile, laugh, or chuckle?  
            3. Punchiness - Is it short, sharp, and scroll-stopping?  
            4. Virality Potential - Would people retweet or share it?  
            5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

            Auto-reject if:
                - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
                - It exceeds 280 characters
                - It reads like a traditional setup-punchline joke
                - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

            ### Respond ONLY in structured format:
                - evaluation: "approved" or "needs_improvement"  
                - feedback: One paragraph explaining the strengths and weaknesses """
        )
    ]

    response = structured_evaluator_llm.invoke(messages)

    return{
        'evaluation': response.evaluation,
        'feedback': response.feedback,
        'feedback_history': [response.feedback]
    }

def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""Improve the tweet based on this feedback: "{state['feedback']}"
            Topic: "{state['topic']}"
            Original Tweet: {state['tweet']}

            Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters."""
        )
    ]

    response = optimizer_llm.invoke(messages).content
    iteration_count = state["iteration"] + 1

    return{
        'tweet': response,
        'iteration': iteration_count,
        'tweet_history': [response]
    }

def route_evaluation(state: TweetState):
    if state['evaluation'] == "approved" or state["iteration"] >= state["max_iteration"]:
        return "approved"
    else:
        return "needs_improvement"

# build graph
graph = StateGraph(TweetState)

# add nodes
graph.add_node('generate', generate_tweet)
graph.add_node('evaluate', evaluate_tweet)
graph.add_node('optimize', optimize_tweet)

# add edges
graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

graph.add_conditional_edges(
    "evaluate", route_evaluation,
    {
        "approved": END,
        "needs_improvement": "optimize"
    }
)

# looping condition
graph.add_edge("optimize", "evaluate")

# compile graph
post_generator_workflow = graph.compile()

initial_state = {
    "topic": "Unemployment In India",
    "iteration": 1,
    "max_iteration": 5,
}

final_state = post_generator_workflow.invoke(initial_state)
print(f"res: {final_state}")

# get your graph image
png_data = post_generator_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "post_generator_workflow.png", "wb") as f:
    f.write(png_data)

print(post_generator_workflow.get_graph().draw_mermaid())


