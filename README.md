# 🤖 AI Powered LangChain Agent for Research Tasks

---

## 📌 Project Overview

This project is a **research assistant tool** powered by **Google Gemini AI** and **LangChain**. It intelligently processes user questions, performs external searches (web + Wikipedia), and returns a structured summary with sources and metadata.

> Ideal for building AI tools, education bots, or information summarizers using LangChain and Gemini.

---

## 🧰 Tech Stack

| Tool/Library               | Purpose                                              |
|---------------------------|------------------------------------------------------|
| `LangChain`               | Framework for chaining LLM tools and agents          |
| `Google Gemini API`       | Large Language Model (flash-lite used)               |
| `langchain-google-genai`  | Gemini integration with LangChain                    |
| `DuckDuckGoSearchRun`     | Tool to perform web searches                         |
| `WikipediaAPIWrapper`     | Wikipedia query tool                                 |
| `Pydantic`                | Schema validation and structured output              |
| `python-dotenv`           | Secure API key management using `.env`               |

---

## 🧠 Functionality

- Accepts a **natural language query**
- Decides if **external tools** are needed via an **agent**
- Queries the **web** and **Wikipedia**
- Structures the response using **Pydantic**
- Returns:
  - A short summary of the topic
  - Sources consulted
  - Tools used
  - Original topic/query


