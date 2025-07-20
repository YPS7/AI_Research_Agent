from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain.tools import DuckDuckGoSearchRun, Tool

# search = DuckDuckGoSearchRun()
# search_tool  =Tool(
#     name="search",
#     func=search.run,
#     description="Search the web for information",
# )

# Instantiate the DuckDuckGo “run” tool
search_runner = DuckDuckGoSearchRun()

# Wrap it in a LangChain Tool
search_tool = Tool(
    name="web_search",
    func=search_runner.run,
    description="Search the web for up-to-date information on a topic.",
)