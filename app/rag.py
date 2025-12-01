import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# 1. LOAD SYSTEM PROMPT FROM FILE

def load_prompt(path: str) -> str:
    """Load a text prompt from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

system_prompt = load_prompt("prompts/system_prompt.txt")

# 2. Initialise Models

llm = AzureChatOpenAI(
    model=os.getenv("CHAT_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
embeddings = AzureOpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

# 3. VECTOR STORE

persist_directory = "chroma_langchain_db"
collection_name = "hr_rag"

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 4. RETRIEVER TOOL

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches, and returns the most relevant sections of information from the employee handbook document
    that is relevant to the employee’s question.
    Focus on policies, procedures, definitions, entitlements, and rules.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the employee handbook document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]
tools_dict = {t.name: t for t in tools}
llm = llm.bind_tools(tools)

# 5. LANGGRAPH AGENT

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {"messages": results}

# 6. BUILD LANGGRAPH

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges("llm", should_continue, {
    True: "retriever_agent",
    False: END
})

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def run_rag():
    conversation = []  # persistent buffer

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # append new user message
        conversation.append(HumanMessage(content=user_input))

        # send ENTIRE conversation history
        result = rag_agent.invoke({"messages": conversation})

        # append model response so it's remembered
        conversation.append(result["messages"][-1])

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)

# To test
# def running_agent():
#     print("\n=== RAG AGENT===")
#     conversation = []
#
#     while True:
#         user_input = input("\nWhat is your question: ")
#         if user_input.lower() in ['exit', 'quit']:
#             break
#
#         # append user's message
#         conversation.append(HumanMessage(content=user_input))
#
#         # run the agent with full conversation history
#         result = rag_agent.invoke({"messages": conversation})
#
#         # append assistant response
#         conversation.append(result["messages"][-1])
#
#         print("\n=== ANSWER ===")
#         print(result['messages'][-1].content)
#
# running_agent()