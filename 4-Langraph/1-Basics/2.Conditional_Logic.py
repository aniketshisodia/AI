# 02_conditional_graph.py
"""
BASIC CONDITIONAL: Graph with a decision point
Flow: Start → Check → (if long) → Process Long → End
                    ↘ (if short) → Process Short → End
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# Define what data we'll track
class State(TypedDict):
    text: str
    length_category: str
    processed: str

def check_length(state: State) -> State:
    """Check if text is long or short"""
    text = state["text"]
    length = len(text)
    
    print(f"📏 Text has {length} characters")
    
    if length > 20:
        category = "long"
    else:
        category = "short"
    
    return {
        "text": state["text"],
        "length_category": category,
        "processed": ""
    }

def process_long(state: State) -> State:
    """Process long text (summarize)"""
    text = state["text"]
    print(f"📚 Processing LONG text...")
    
    # Simple processing (mock)
    processed = f"Summary of {len(text)} chars: {text[:50]}..."
    
    return {
        "text": state["text"],
        "length_category": state["length_category"],
        "processed": processed
    }

def process_short(state: State) -> State:
    """Process short text (capitalize)"""
    text = state["text"]
    print(f"✨ Processing SHORT text...")
    
    # Simple processing
    processed = text.upper()
    
    return {
        "text": state["text"],
        "length_category": state["length_category"],
        "processed": processed
    }

# CONDITIONAL FUNCTION - decides which path to take
def decide_route(state: State) -> Literal["long", "short"]:
    """Return next node name based on state"""
    if state["length_category"] == "long":
        print("🔀 Routing to LONG processor")
        return "long"
    else:
        print("🔀 Routing to SHORT processor")
        return "short"

# Build the graph
workflow = StateGraph(State)

# Add all nodes
workflow.add_node("check", check_length)
workflow.add_node("long", process_long)
workflow.add_node("short", process_short)

# Set entry point
workflow.set_entry_point("check")

# Add conditional edge
workflow.add_conditional_edges(
    "check",           # From this node
    decide_route,      # Use this function to decide
    {
        "long": "long",    # If returns "long" → go to "long" node
        "short": "short"   # If returns "short" → go to "short" node
    }
)

# Add edges to end
workflow.add_edge("long", END)
workflow.add_edge("short", END)

# Compile
app = workflow.compile()

# Test with different inputs
print("="*50)
print("TEST 1: Short text")
print("="*50)
result1 = app.invoke({"text": "Hello world", "length_category": "", "processed": ""})
print(f"\n✅ Result: {result1['processed']}\n")

print("="*50)
print("TEST 2: Long text")
print("="*50)
result2 = app.invoke({"text": "This is a very long sentence that definitely has more than twenty characters in it", 
                      "length_category": "", "processed": ""})
print(f"\n✅ Result: {result2['processed']}")