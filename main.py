
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.prompts.chat import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from langchain_community.utilities import WikipediaAPIWrapper
load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env file")
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    google_api_key=GEMINI_KEY,
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a research assistant. Answer the user query using tools as needed.
        Return *only* valid JSON matching the schema:
        {format_instructions}
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())
ddg = DuckDuckGoSearchRun()
search_tool = Tool(
    name="web_search",
    func=ddg.run,
    description="Search the web for up-to-date information.",
)

wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="wiki_search",
    func=wiki.run,
    description="Lookup Wikipedia articles.",
)

tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
if __name__ == "__main__":
    query = input("What can I help you with? ")
    raw = agent_executor.invoke({"query": query})

    try:
        txt = raw["output"]
        clean = txt.lstrip("```json").rstrip("```").strip()
        structured: ResearchResponse = parser.parse(clean)
        print(f"Query: {query}")
        print(f"Answer: {structured.summary}")
        print(f"Sources: {', '.join(structured.sources)}")
        print(f"Tools Used: {', '.join(structured.tools_used)}")
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw agent output:", raw)
