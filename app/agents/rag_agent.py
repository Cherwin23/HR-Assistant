"""
RAG Agent
LangGraph agent setup for RAG processing with tool support.
"""
import asyncio
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from app.config.settings import CHAT_MODEL, AZURE_OPENAI_API_VERSION, SYSTEM_PROMPT_PATH
from app.utils.prompt_loader import load_prompt
from app.tools.handbook_tool import handbook_retriever_tool
from app.tools.employee_tool import employee_data_sql_tool

# Load system prompt
system_prompt = load_prompt(SYSTEM_PROMPT_PATH)

# Initialize LLM
llm = AzureChatOpenAI(
    model=CHAT_MODEL,
    api_version=AZURE_OPENAI_API_VERSION,
)

# Bind tools to LLM
tools = [handbook_retriever_tool, employee_data_sql_tool]
tools_dict = {t.name: t for t in tools}
llm = llm.bind_tools(tools)


# LangGraph Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


async def call_llm(state: AgentState) -> AgentState:
    """Async function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = await llm.ainvoke(messages)
    return {"messages": [message]}


async def _execute_single_tool(tool_call: dict) -> tuple:
    """Execute a single tool call asynchronously. Returns (tool_call_id, tool_name, result)."""
    tool_name = tool_call['name']
    tool_call_id = tool_call['id']
    args = tool_call.get('args', {})
    
    # Get the appropriate argument for logging
    if 'query' in args:
        arg_value = args['query']
        print(f"Calling Tool: {tool_name} with query: {arg_value}")
    elif 'sql_query' in args:
        arg_value = args['sql_query']
        print(f"Calling Tool: {tool_name} with SQL: {arg_value}")
    else:
        arg_value = str(args)
        print(f"Calling Tool: {tool_name} with args: {arg_value}")

    if tool_name not in tools_dict:
        print(f"\nTool: {tool_name} does not exist.")
        result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
    else:
        tool = tools_dict[tool_name]
        
        # For blocking operations like SQLite, use asyncio.to_thread
        # For LangChain tools that support async, use ainvoke
        if tool_name == "employee_data_sql_tool":
            # Run blocking SQLite operation in a thread pool
            result = await asyncio.to_thread(tool.invoke, args)
        else:
            # Try async invoke first, fallback to sync in thread if needed
            try:
                result = await tool.ainvoke(args)
            except AttributeError:
                # Fallback to synchronous invoke in thread if ainvoke not available
                result = await asyncio.to_thread(tool.invoke, args)
    
    return (tool_call_id, tool_name, str(result))


async def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response in parallel using asyncio."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    
    # Execute tools in parallel using asyncio.gather
    if len(tool_calls) > 1:
        print(f"[PARALLEL] Executing {len(tool_calls)} tools concurrently with asyncio...")
        # Execute all tools in parallel - asyncio.gather maintains order
        tool_results = await asyncio.gather(*[
            _execute_single_tool(t) for t in tool_calls
        ])
        
        # Build results in original order (asyncio.gather maintains order)
        for tool_call_id, tool_name, result in tool_results:
            results.append(ToolMessage(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=result
            ))
    else:
        # Single tool call - execute directly
        tool_call_id, tool_name, result = await _execute_single_tool(tool_calls[0])
        results.append(ToolMessage(
            tool_call_id=tool_call_id,
            name=tool_name,
            content=result
        ))
    
    return {"messages": results}


# Build LangGraph
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