from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain.tools import DuckDuckGoSearchRun, Tool

search_runner = DuckDuckGoSearchRun()
search_tool = Tool(
    name="web_search",
    func=search_runner.run,
    description="Search the web for up-to-date information on a topic.",
)
