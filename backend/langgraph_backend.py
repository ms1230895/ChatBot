from dotenv import load_dotenv
import os
import uuid
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_anthropic import ChatAnthropic


load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv("Anthropic_API_KEY", "")

model = ChatAnthropic(
    model="claude-sonnet-5-5",
    max_tokens=1000,
)


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    # take user query from state
    messages = state['messages']

    # send to llm
    response = model.invoke(messages)

    # response store state
    return {'messages': [response]}


# checkpointer keeps conversation history per thread_id
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)


def _config(thread_id: str):
    return {'configurable': {'thread_id': thread_id}}


def new_thread_id() -> str:
    # unique id per chat session, so different browser tabs don't share history
    return str(uuid.uuid4())


def get_history(thread_id: str) -> list[dict]:
    # read past messages from the checkpointer and convert to {'role', 'content'} for the UI
    messages = chatbot.get_state(_config(thread_id)).values.get('messages', [])
    return [
        {'role': 'user' if isinstance(m, HumanMessage) else 'assistant', 'content': m.text}
        for m in messages
    ]


def stream_response(user_input: str, thread_id: str):
    # stream_mode='messages' yields (message_chunk, metadata) token by token
    for message_chunk, _ in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=_config(thread_id),
        stream_mode='messages',
    ):
        # .text skips thinking/signature blocks, only yields reply text
        if isinstance(message_chunk, AIMessageChunk) and message_chunk.text:
            yield message_chunk.text
