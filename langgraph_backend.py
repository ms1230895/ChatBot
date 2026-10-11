from dotenv import load_dotenv
import os
import uuid
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv("Anthropic_API_KEY", "")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HuggingFace_API_KEY", "")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    task="text-generation",
)

model2 = ChatHuggingFace(
    llm=llm,
    max_tokens=1000,
    temperature=0.7,
)

model = ChatAnthropic(
    model="claude-sonnet-5-5",
    max_tokens=1000,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
#defien the nodes of the graph
graph.add_node("chat_node", chat_node)
#Define the edges of the graph
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)