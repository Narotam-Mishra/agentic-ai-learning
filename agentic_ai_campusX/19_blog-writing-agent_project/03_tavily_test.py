
# tavily test

from dotenv import load_dotenv
from langchain_tavily import TavilySearch


load_dotenv(override=True)

tool = TavilySearch(max_results=2)
results = tool.invoke({"query": "ChatGPT version releases and updates from 2022 to 2026"})
print(f"results: {results}")