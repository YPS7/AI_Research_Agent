import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

from tools import search_tool, wiki_tool, deep_research_tool

load_dotenv()
OPENROUTER_KEY = os.getenv("API_KEY")
if not OPENROUTER_KEY:
    raise RuntimeError("Please set API_KEY in your .env (your OpenRouter key)")
print("OpenRouter key loaded successfully.")  

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENROUTER_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
)


plan_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research planner. Given a research question, output a JSON list of steps "
     "using only these tools: web_search, wiki_search, deep_research. "
     "Format exactly as: "
     "[{{\"tool\":\"tool_name\",\"input\":\"optional input\"}}, ...]"),
    ("human", "{query}"),
])
plan_chain = LLMChain(llm=llm, prompt=plan_prompt, output_key="plan", verbose=True)


def execute_steps(plan: List[Dict[str, Any]], query: str):
    summaries = []
    all_sources = []
    for step in plan:
        tool_name = step["tool"]
        inp = step.get("input", query)

        if tool_name == "web_search":
            raw = search_tool.func(inp)
            content, sources = raw, []
        elif tool_name == "wiki_search":
            raw = wiki_tool.func(inp)
            content, sources = raw, []
        elif tool_name == "deep_research":
            raw = deep_research_tool.func(inp)
            content = raw.get("analysis", "No analysis available")
            sources = raw.get("sources", [])
        else:
            continue

        if not content:
            content = "Tool returned no content."  

        summary = llm.invoke(f"Summarize the following result:\n{content}").content
        summaries.append(summary)
        all_sources.extend(sources)

    return {"summaries": summaries, "sources": list(set(all_sources))}


synth_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert researcher. Combine these summaries into a coherent report "
     "with proper citations."),
    ("human", "Summaries:\n{summaries}\nSources:\n{sources}")
])
synth_chain = LLMChain(llm=llm, prompt=synth_prompt, output_key="report", verbose=True)

if __name__ == "__main__":
    query = input("What can I help you with? ")

    # A) Plan
    plan_result = plan_chain.invoke({"query": query})
    plan_str = plan_result["plan"] if isinstance(plan_result, dict) else plan_result
    try:
        plan = json.loads(plan_str)
    except json.JSONDecodeError:
        print("Planning output was not valid JSON:\n", plan_str)

        plan_str = plan_str.strip().replace("'", '"')
        try:
            plan = json.loads(plan_str)
        except json.JSONDecodeError:
            exit(1)

    # B) Execute
    results = execute_steps(plan, query)

    # C) Synthesize
    final_report = synth_chain.invoke({
        "summaries": "\n\n".join(results["summaries"]),
        "sources": "\n".join(results["sources"]),
    })

    print("\n=== Final Deep Research Report ===\n")
    print(final_report['report'])  
