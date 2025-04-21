from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

tavily_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

tools = [
    Tool(
        name="Tavily Web Search",
        func=tavily_tool.run,
        description="Useful for searching real-time web results."
    )
]

# Research Agent
research_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
