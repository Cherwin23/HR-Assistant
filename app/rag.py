import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from operator import add as add_messages
import asyncio
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from prompts.prompt_loader import load_prompt
from app.intent_classifier import classify_intent
from app.employee_data import EMPLOYEE_DB_PATH, run_employee_sql
from app.employee_schema import Employee_Schema_Description

load_dotenv()

# 1. Load System Prompt
prompt = load_prompt("prompts/system_prompt.txt")

# 2. Initialise Models
llm = AzureChatOpenAI(
    model=os.getenv("CHAT_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
embeddings = AzureOpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

# 3. Vector Store and Retriever
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

@tool
def handbook_retriever_tool(query: str) -> str:
    """
    Search the Employee Handbook for policies, procedures, definitions, entitlements, and rules.
    Returns the top relevant sections.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the employee handbook."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

# 4. Employee Data

@tool
def employee_data_sql_tool(sql_query: str) -> str:
    """
    Execute a read-only SQL query against the employee data (SQLite).
    Only SELECT statements are allowed. Table name: employees
    Schema is injected at runtime.
    """
    return run_employee_sql(sql_query, db_path=EMPLOYEE_DB_PATH)

employee_data_sql_tool.description = f"""
Execute a read-only SQL query against the employee data (SQLite).
- Only SELECT statements are allowed.
- Table name: employees
- Use exact SQL, not natural language.

Schema:
{Employee_Schema_Description}
"""

tools = [handbook_retriever_tool, employee_data_sql_tool]
tools_dict = {t.name: t for t in tools}
llm = llm.bind_tools(tools)

# 5. LangGraph Agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

async def call_llm(state: AgentState) -> AgentState:
    """Async function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=prompt)] + messages
    message = await llm.ainvoke(messages)  # Use ainvoke for async
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

# 6. Build Graph
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

# 7. Intent-Aware RAG Agent
async def process_with_intent_classification(
    question: str, 
    conversation_history: Optional[List[BaseMessage]] = None
) -> Dict[str, Any]:
    """
    Process user question with intent classification and conditional RAG.
    Returns response matching schema in Section 5.1.
    
    Args:
        question: User's input question
        conversation_history: Optional list of previous messages for context
    
    Returns:
        Dictionary matching Section 5.1 response schema:
        {
            "intent": str,
            "category": str,
            "module": str | None,
            "use_case": str | None,
            "answer": str | None,
            "confidence": float,
            "requires_context": List[str],
            "entities": Dict[str, Any]
        }
    """
    # Step 1: Intent Classification
    intent_result = classify_intent(question, conversation_history)
    
    category = intent_result.get("category", "").lower()
    confidence = intent_result.get("confidence", 0.0)
    
    # Step 2: Category-Based Routing
    
    # INVALID or low confidence: Return rejection
    if category == "invalid" or confidence < 0.6:
        return {
            "intent": intent_result.get("intent", "invalid"),
            "category": "invalid",
            "module": intent_result.get("module"),
            "use_case": intent_result.get("use_case"),
            "answer": intent_result.get("answer", "I can only assist with HR-related queries. Please rephrase your question."),
            "confidence": confidence,
            "requires_context": [],
            "entities": intent_result.get("entities", {})
        }
    
    # CONVERSATIONAL: Return pre-generated answer
    if category == "conversational":
        return {
            "intent": intent_result.get("intent", "conversational"),
            "category": "conversational",
            "module": None,
            "use_case": None,
            "answer": intent_result.get("answer", "Hello! How can I help you with HR-related questions today?"),
            "confidence": confidence,
            "requires_context": [],
            "entities": {}
        }
    
    # ACTION: Return intent classification result (answer will be null, Gateway handles it)
    if category == "action":
        return {
            "intent": intent_result.get("intent"),
            "category": "action",
            "module": intent_result.get("module"),
            "use_case": intent_result.get("use_case"),
            "answer": None,  # Gateway will handle action processing
            "confidence": confidence,
            "requires_context": intent_result.get("requires_context", []),
            "entities": intent_result.get("entities", {})
        }
    
    # QUERY: Use RAG to generate answer
    if category == "query":
        # Prepare messages for RAG
        if conversation_history:
            rag_messages = list(conversation_history)
        else:
            rag_messages = []
        
        # Add current question
        rag_messages.append(HumanMessage(content=question))
        
        # Run RAG agent asynchronously
        try:
            rag_result = await rag_agent.ainvoke({"messages": rag_messages})
            rag_answer = rag_result["messages"][-1].content
            
            return {
                "intent": intent_result.get("intent"),
                "category": "query",
                "module": intent_result.get("module"),
                "use_case": intent_result.get("use_case"),
                "answer": rag_answer,
                "confidence": confidence,
                "requires_context": [],
                "entities": intent_result.get("entities", {})
            }
        except Exception as e:
            print(f"RAG error: {e}")
            # Fallback: return intent classification with error message
            return {
                "intent": intent_result.get("intent"),
                "category": "query",
                "module": intent_result.get("module"),
                "use_case": intent_result.get("use_case"),
                "answer": "I encountered an error retrieving information. Please try again.",
                "confidence": confidence * 0.5,  # Reduce confidence due to error
                "requires_context": [],
                "entities": intent_result.get("entities", {})
            }
    
    # Fallback (should not reach here)
    return {
        "intent": intent_result.get("intent", "invalid"),
        "category": "invalid",
        "module": None,
        "use_case": None,
        "answer": "I encountered an error processing your request. Please try again.",
        "confidence": 0.0,
        "requires_context": [],
        "entities": {}
    }

async def run_rag():
    conversation = []  # persistent buffer

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # append new user message
        conversation.append(HumanMessage(content=user_input))

        # send ENTIRE conversation history
        result = await rag_agent.ainvoke({"messages": conversation})

        # append model response so it's remembered
        conversation.append(result["messages"][-1])

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


# Main entry point for CLI usage
if __name__ == "__main__":
    asyncio.run(run_rag())