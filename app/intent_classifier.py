import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from prompts.prompt_loader import load_prompt

load_dotenv()

# 1. Load Intent Classification Prompt
prompt = load_prompt("prompts/intent_classification_prompt.txt")

# 2. Initialise Model
llm = AzureChatOpenAI(
    model=os.getenv("CHAT_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.1,  # Lower temperature for more consistent classification
)

# JSON parser for structured output
json_parser = JsonOutputParser()

def classify_intent(question: str, conversation_history: Optional[list] = None) -> Dict[str, Any]:
    """
    Classify user intent and extract entities.
    
    Args:
        question: User's input question
        conversation_history: Optional list of previous messages for context
    
    Returns:
        Dictionary with intent, category, module, use_case, answer, confidence, 
        requires_context, and entities
    """
    # Build messages
    messages = [SystemMessage(content=prompt)]
    
    # Add conversation history if available (last 3 messages for context)
    if conversation_history:
        # Include last few messages for context
        context_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        messages.extend(context_messages)
    
    # Add current question
    messages.append(HumanMessage(content=f"Classify this user input:\n\n{question}"))
    
    try:
        # Get response from LLM
        response = llm.invoke(messages)
        
        # Parse JSON from response content
        content = response.content
        
        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        # Parse JSON
        result = json.loads(content)
        
        # Validate and normalize response structure
        normalized = normalize_intent_response(result)
        
        return normalized
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response content: {response.content}")
        # Return default invalid response
        return {
            "intent": "invalid",
            "category": "invalid",
            "module": None,
            "use_case": None,
            "answer": "I encountered an error processing your request. Please try rephrasing your question.",
            "confidence": 0.0,
            "requires_context": [],
            "entities": {}
        }
    except Exception as e:
        print(f"Intent classification error: {e}")
        return {
            "intent": "invalid",
            "category": "invalid",
            "module": None,
            "use_case": None,
            "answer": "I encountered an error processing your request. Please try again.",
            "confidence": 0.0,
            "requires_context": [],
            "entities": {}
        }

def normalize_intent_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate intent classification response.
    Ensures all required fields are present with correct types.
    """
    # Default entity structure
    default_entities = {
        "days": None,
        "leave_type": None,
        "start_date": None,
        "end_date": None,
        "department": None,
        "role": None,
        "location": None,
        "name": None,
        "employee_id": None,
        "job_family": None
    }
    
    # Merge provided entities with defaults
    entities = result.get("entities", {})
    normalized_entities = {**default_entities, **entities}
    
    # Ensure confidence is a float between 0 and 1
    confidence = result.get("confidence", 0.5)
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))
    
    # Normalize category to lowercase
    category = result.get("category", "invalid").lower()
    if category not in ["query", "action", "conversational", "invalid"]:
        category = "invalid"
    
    # Normalize module
    module = result.get("module")
    if module and module not in ["M1", "M2", "M3"]:
        module = None
    
    # Build normalized response
    normalized = {
        "intent": result.get("intent", "invalid"),
        "category": category,
        "module": module,
        "use_case": result.get("use_case"),
        "answer": result.get("answer"),  # Can be None for action intents
        "confidence": confidence,
        "requires_context": result.get("requires_context", []),
        "entities": normalized_entities
    }
    
    return normalized