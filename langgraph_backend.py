from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama


# -----------------------------
# Local Ollama Model
# -----------------------------
model = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)
# -----------------------------
# State
# -----------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# Chat Node
# -----------------------------
def chat_node(state: ChatState):
    messages = state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [response]
    }


# -----------------------------
# Memory
# -----------------------------
checkpointer = InMemorySaver()


# -----------------------------
# Graph
# -----------------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# -----------------------------
# Compile
# -----------------------------
chatbot = graph.compile(
    checkpointer=checkpointer
)