
# upsc eassy parallel workflow

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from IPython.display import Image, display
from pathlib import Path
import operator
import json

load_dotenv(override=True)

llm_model = ChatOpenAI(
    model='gpt-4o-mini'
)

class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the eassy')
    score: int = Field(description='Score out of 10', ge=0, le=10)

structured_model = llm_model.with_structured_output(EvaluationSchema)

eassy = """
India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society.
"""

prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {eassy}"

res = structured_model.invoke(prompt)
# print(f"res: {res}")
# print(f"Score: {res.score}")
# print(f"Feedback: {res.feedback}")

# define state of the workflow
class UPSCState(TypedDict):
    eassy: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

def evaluate_language(state: UPSCState):
    prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state['eassy']}"
    eval_lang_output = structured_model.invoke(prompt)

    return {
        'language_feedback': eval_lang_output.feedback,
        'individual_scores': [eval_lang_output.score]
    }



def evaluate_analysis(state: UPSCState):
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state['eassy']}"
    anaysis_feedback_output = structured_model.invoke(prompt)

    return {
        'analysis_feedback': anaysis_feedback_output.feedback,
        'individual_scores': [anaysis_feedback_output.score]
    }

def evaluate_thought(state: UPSCState):
    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state['eassy']}"
    evaluate_thought_output = structured_model.invoke(prompt)

    return {
        'clarity_feedback': evaluate_thought_output.feedback,
        'individual_scores': [evaluate_thought_output.score]
    }

def final_evaluation(state: UPSCState):
    # summary feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = llm_model.invoke(prompt).content

    # average calculation of all scores
    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])

    return {
        'overall_feedback': overall_feedback,
        'avg_score': avg_score
    }


# define graph
graph = StateGraph(UPSCState)

# add nodes
graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# add edges
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')


graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')

graph.add_edge('final_evaluation', END)

# compile the graph
eassy_workflow = graph.compile()

initial_state = {
    'eassy': eassy
}

final_state = eassy_workflow.invoke(initial_state)

terminal_output = {
    "average_score": final_state["avg_score"],
    "individual_scores": final_state["individual_scores"],
    "feedback": {
        "language": final_state["language_feedback"],
        "analysis": final_state["analysis_feedback"],
        "clarity": final_state["clarity_feedback"],
        "overall": final_state["overall_feedback"],
    },
}

print("\nFinal Result:")
print(json.dumps(terminal_output, indent=4, ensure_ascii=False))

# get your graph image
png_data = eassy_workflow.get_graph().draw_mermaid_png()
display(Image(png_data))

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)

with open(images_dir / "eassy_workflow.png", "wb") as f:
    f.write(png_data)

print(eassy_workflow.get_graph().draw_mermaid())
