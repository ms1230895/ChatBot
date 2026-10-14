# LangGraph Chatbot

A multi-utility chatbot built with **LangGraph**, **LangChain** and **Streamlit**, running on a local **Ollama** model (`llama3.1`).

The project is built up in stages. Each stage adds a feature on top of the previous one: streaming, conversation sidebar, SQLite persistence, tool calling, MCP servers and RAG over PDFs.

## Features

- **Local LLM** – runs `llama3.1:latest` through Ollama. No paid API key needed for the model.
- **Streaming responses** – tokens stream into the Streamlit chat UI as they are generated.
- **Multiple conversations** – sidebar with "New Chat" and a list of past threads (one `thread_id` per chat).
- **Persistent memory** – chat history is checkpointed to `chatbot.db` (SQLite) with LangGraph's `SqliteSaver`, so conversations survive restarts.
- **Tool calling** – the agent can call:
  - DuckDuckGo web search
  - Calculator (`add`, `sub`, `mul`, `div`)
  - Stock price lookup (Alpha Vantage `GLOBAL_QUOTE`)
- **MCP integration** – loads extra tools from MCP servers via `langchain-mcp-adapters` (stdio and streamable HTTP).
- **RAG over PDFs** – upload a PDF per chat; it is chunked, embedded with `nomic-embed-text` and indexed in FAISS. The agent answers questions about it through a `rag_tool`.

## Architecture

```
Streamlit UI  ──►  LangGraph StateGraph  ──►  ChatOllama (llama3.1)
                     │        ▲
                     ▼        │  tools_condition
                  ToolNode ───┘  (search, calculator, stock, MCP, RAG)
                     │
                     ▼
               SqliteSaver / AsyncSqliteSaver  ──►  chatbot.db
```

The graph is `START → chat_node`. If the LLM requests a tool, `tools_condition` routes to the `tools` node, which loops back to `chat_node`. Otherwise the graph ends.

## Project structure

| Stage | Backend | Frontend | What it adds |
|---|---|---|---|
| 1. Basic chat | `langgraph_backend.py` | `streamlit_frontend.py` | Single-thread chat, in-memory checkpointer |
| 2. Streaming | `langgraph_backend.py` | `streamlit_frontend_streaming.py` | Token streaming with `st.write_stream` |
| 3. Sidebar / resume | `langgraph_backend.py` | `streamlit_frontend_Resume.py` | New Chat button, switch between threads |
| 4. Database | `langgraph_backend_database.py` | `streamlit_frontend_database.py` | SQLite persistence of all threads |
| 5. Tools | `langgraph_tools_backend.py` | `streamlit_frontend_tool.py` | Search, calculator, stock price tools with live tool status |
| 6. MCP | `langgraph_mcp_backend.py` | `streamlit-frontend-mcp.py` | Async graph + MCP server tools |
| 7. RAG | `langgraph_backend_Rag.py` | `Streamlit-frontend-Rag.py` | PDF upload, FAISS retriever, all tools combined |

## Prerequisites

- Python 3.10+ (developed on 3.13)
- [Ollama](https://ollama.com) installed and running
- The models used by the project:

```bash
ollama pull llama3.1:latest
ollama pull nomic-embed-text:latest   # only needed for the RAG stage
```

## Installation

```bash
git clone https://github.com/ms1230895/ChatBot---LangGraph.git
cd ChatBot---LangGraph

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install streamlit langgraph langgraph-checkpoint-sqlite langchain-core \
    langchain-community langchain-ollama langchain-openai langchain-text-splitters \
    langchain-mcp-adapters duckduckgo-search faiss-cpu pypdf aiosqlite \
    requests python-dotenv
```

Optional: create a `.env` file for environment variables (it is git-ignored). The backends load it with `python-dotenv`.

## Usage

Run the frontend for the stage you want. The latest, most complete version is the RAG chatbot:

```bash
streamlit run Streamlit-frontend-Rag.py
```

Other stages:

```bash
streamlit run streamlit_frontend_database.py   # persistent chat
streamlit run streamlit_frontend_tool.py       # chat + tools
streamlit run streamlit-frontend-mcp.py        # chat + MCP tools
```

Then open the URL Streamlit prints (default http://localhost:8501).

### Using the RAG chatbot

1. Click **New Chat** (or use the current thread).
2. Upload a PDF from the sidebar. Wait for "✅ PDF indexed".
3. Ask questions about the document. You can also ask for web searches, calculations or stock prices in the same chat.

> The PDF index is kept in memory per thread. It is lost when the app restarts, while the chat history stays in `chatbot.db`.

## Configuration notes

- **Model** – change `model="llama3.1:latest"` in the backend files to use another Ollama model. The model must support tool calling for stages 5–7.
- **Stock tool** – uses an Alpha Vantage API key. Get a free key at https://www.alphavantage.co and set it in the `get_stock_price` tool.
- **MCP servers** – edit the `MultiServerMCPClient` config in `langgraph_mcp_backend.py`. Point the `arith` server `args` to the path of your own MCP server script. If no MCP server can be reached, the app still starts with the built-in tools.
- **Imports in stages 1–3** – `streamlit_frontend.py`, `streamlit_frontend_streaming.py` and `streamlit_frontend_Resume.py` import `Chatbot.langgraph_backend`. If you get a `ModuleNotFoundError`, change the import to `from langgraph_backend import chatbot`.

## Tech stack

- [LangGraph](https://github.com/langchain-ai/langgraph) – agent graph, checkpointing
- [LangChain](https://github.com/langchain-ai/langchain) – LLM, tools, loaders, vector store
- [Ollama](https://ollama.com) – local LLM and embeddings
- [Streamlit](https://streamlit.io) – chat UI
- [FAISS](https://github.com/facebookresearch/faiss) – vector search
- SQLite – conversation persistence

## Author

Gaurav Gawai – [@ms1230895](https://github.com/ms1230895)
