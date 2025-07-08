from phi.agent import Agent
from phi.model.groq import Groq 
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key=os.getenv("OPENAI_API_KEY")


websearch_agent = Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-70b-8192"),
    tools = [DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name = " Finance AI Agent",
    model = Groq(id="llama3-70b-8192"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news = True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agents = Agent(
    model = Groq(id="llama3-70b-8192"),
    team = [websearch_agent, finance_agent],
    instructions=["Always include instructions", "Use table to display data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agents.print_response("Summarize analyst recommendation and share latest news for NVDA", stream=True)