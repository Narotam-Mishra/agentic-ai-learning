
# using LangSmith with AI Agent

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_react_agent
from langsmith import Client
import os

load_dotenv(override=True)

weather_api_key = os.getenv('WEATHER_API_KEY')

os.environ['LANGCHAIN_PROJECT'] = 'ReAct_Agent'

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f"https://api.weatherstack.com/current?access_key={weather_api_key}&query={city}"
  response = requests.get(url)

  return response.json()

llm = ChatOpenAI()

# Step 2: Pull the ReAct prompt from LangChain Hub
client = Client()
prompt = client.pull_prompt("hwchase17/react")

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "Identify the birthplace city of Kalpana Chawla (search) and give its current temperature."})
print(response)

print(response['output'])