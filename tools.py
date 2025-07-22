import os
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from firecrawl import FirecrawlApp


ddg_runner = DuckDuckGoSearchRun()
search_tool = Tool(
    name="web_search",
    func=ddg_runner.run,
    description="Search the web for up-to-date information.",
)


wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="wiki_search",
    func=wiki.run,
    description="Lookup Wikipedia articles.",
)


def deep_research(query: str, max_depth: int = 3, time_limit: int = 180, max_urls: int = 10, retries: int = 3):
    """
    Perform a deeper web crawl + analysis via Firecrawl with retries.
    Expects FIRECRAWL_API_KEY in your .env.
    """
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    params = {"maxDepth": max_depth, "timeLimit": time_limit, "maxUrls": max_urls}
    for attempt in range(retries):
        try:
            result = app.deep_research(query=query, params=params)
            if result.get("data") and result["data"].get("finalAnalysis"):  
                return {
                    "analysis": result["data"]["finalAnalysis"],
                    "sources": result["data"]["sources"],
                }
            else:
                print(f"Empty response from Firecrawl on attempt {attempt + 1}. Retrying...")
        except Exception as e:
            print(f"Firecrawl error on attempt {attempt + 1}: {str(e)}. Retrying...")
    return {"analysis": "Deep research failed after retries.", "sources": []}  

deep_research_tool = Tool(
    name="deep_research",
    func=deep_research,
    description="Run an in-depth multi-hop web crawl and analysis via Firecrawl."
)
