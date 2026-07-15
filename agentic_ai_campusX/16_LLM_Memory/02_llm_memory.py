
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOpenAI()

messages = ["Hi, My Name is Rahul"]

output = llm.invoke(messages).content

messages.append(output)
messages.append("What is my name?")

output = llm.invoke(messages).content

print(f"output: {output}")
