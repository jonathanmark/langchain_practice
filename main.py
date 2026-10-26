import os
import sys
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for the agent's response."""

    answer: str = Field(description="The answer to the query.")
    sources: List[Source] = Field(default_factory=list, description="A list of sources used to generate the answer.")

llm = ChatDeepSeek(api_key=os.environ.get("DEEPSEEK_API_KEY"), model="deepseek-v4-pro", temperature=0.7)
tools = [TavilySearch(api_key=os.environ.get("TAVILY_API_KEY"))]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages": HumanMessage(content="Please search me some job openings for AI Professionals in the philippines that pay minimum 130k php a month. I prefer companies that are Fortune 500 and have local offices in the Philippines. Please provide me with the job title, company name, location, and salary range for each opening you find. Please provide me with at least 5 job openings.")})
    print(result)
    
if __name__ == "__main__":
    main()