
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

# simple one line prompt
prompt = PromptTemplate.from_template("{question}")

# initialize model and parser
model = ChatOpenAI()
parser = StrOutputParser()

# chain: prompt -> model -> parser
chain = prompt | model | parser

# run it
res = chain.invoke({
    "question": "What is capital of Argentina?"
})

print(f"result: {res}")