from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOllama

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="llama3.2:1b")

def chatbot_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot_node)

graph_builder.set_entry_point("chatbot")
graph_builder.add_edge("chatbot", END)

app = graph_builder.compile()

print("Your LangGraph chatbot is ready. Type your message below:")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = app.invoke({"messages": [("human", user_input)]})
    print(f"Bot: {response['messages'][-1].content}")