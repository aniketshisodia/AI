# 03_looping_graph.py
"""
GRAPH WITH LOOPS: Can retry if quality is bad
Flow: Generate → Check Quality → (if bad) → Generate (again)
                               ↘ (if good) → End
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class State(TypedDict):
    attempt: int
    generated_text: str
    quality_score: int
    max_attempts: int

def generate_text(state: State) -> State:
    """Generate text (simulated with quality based on attempt number)"""
    attempt = state["attempt"]
    print(f"🔄 Attempt #{attempt}")
    
    # Simulate better quality on later attempts
    if attempt == 1:
        text = "bad quality text"
        quality = 30
    elif attempt == 2:
        text = "decent quality text"
        quality = 60
    else:
        text = "excellent quality text with proper content"
        quality = 90
    
    print(f"   Generated: '{text}' (quality: {quality})")
    
    return {
        "attempt": attempt,
        "generated_text": text,
        "quality_score": quality,
        "max_attempts": state["max_attempts"]
    }

def check_quality(state: State) -> Literal["good", "bad"]:
    """Check if quality is acceptable"""
    score = state["quality_score"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]
    
    print(f"   Quality check: {score}/100")
    
    if score >= 70:
        print("   ✅ Quality ACCEPTABLE!")
        return "good"
    elif attempt >= max_attempts:
        print(f"   ⚠️  Max attempts ({max_attempts}) reached, stopping")
        return "good"  # Stop even if not perfect
    else:
        print("   ❌ Quality TOO LOW, retrying...")
        return "bad"

def increment_attempt(state: State) -> State:
    """Prepare for next attempt"""
    return {
        "attempt": state["attempt"] + 1,
        "generated_text": "",
        "quality_score": 0,
        "max_attempts": state["max_attempts"]
    }

# Build the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate", generate_text)
workflow.add_node("increment", increment_attempt)

# Set entry
workflow.set_entry_point("generate")

# Add conditional edge - this creates the LOOP!
workflow.add_conditional_edges(
    "generate",
    check_quality,
    {
        "good": END,           # Good quality → finish
        "bad": "increment"     # Bad quality → try again
    }
)

# From increment, go back to generate
workflow.add_edge("increment", "generate")

# Compile
app = workflow.compile()

# Run with max 3 attempts
print("🚀 Starting quality improvement loop...\n")
result = app.invoke({
    "attempt": 1,
    "generated_text": "",
    "quality_score": 0,
    "max_attempts": 3
})

print(f"\n✅ Final output: '{result['generated_text']}'")
print(f"📊 Total attempts: {result['attempt']}")