
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv(override=True)

# update langchain project name in env file
os.environ['LANGCHAIN_PROJECT'] = 'Sequential_LLM_App'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

model2 = ChatOpenAI(model='gpt-4o', temperature=0.5)

parser = StrOutputParser()

# 2-step sequential chain
chain = prompt1 | model1 | parser | prompt2 | model2 | parser

# config for metadata
config = {
    'run_name': 'sequential_chain',
    'tags': ['llm_app', 'report generation', 'summarization'],
    'metadata': {
        'model1': 'gpt-4o-mini',
        'model1_temp': 0.7,
        'parser': 'stroutputparser'
    },
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)