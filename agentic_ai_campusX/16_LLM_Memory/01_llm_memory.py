
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOpenAI()

res1 = llm.invoke("Hi, My Name is Rahul").content

res2 = llm.invoke("What is my name?").content

print(f"res1: {res1}")
print(f"res2: {res2}")