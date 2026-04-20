# 01_hello_graph.py
"""
ABSOLUTE BASIC: A graph with 2 nodes in sequence
Flow: Start → Node A → Node B → End
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

# STEP 1: Define the State (like a shared notebook)
class State(TypedDict):
    message: str
    counter: int

# STEP 2: Define Node Functions (each does one thing)
def node_a(state: State) -> State:
    """First node - adds 'Hello' to message"""
    print(f"📍 Node A received: {state}")
    
    # Update the state
    return {
        "message": state["message"] + " Hello",
        "counter": state["counter"] + 1
    }

def node_b(state: State) -> State:
    """Second node - adds 'World' to message"""
    print(f"📍 Node B received: {state}")
    
    # Update the state
    return {
        "message": state["message"] + " World",
        "counter": state["counter"] + 1
    }

# STEP 3: Build the Graph
workflow = StateGraph(State)  # Create empty graph with our State type

# Add nodes to the graph
workflow.add_node("first", node_a)      # Name it "first", use node_a function
workflow.add_node("second", node_b)     # Name it "second", use node_b function

# Define the flow (edges)
workflow.set_entry_point("first")       # Start here
workflow.add_edge("first", "second")    # After first, go to second
workflow.add_edge("second", END)        # After second, finish

# STEP 4: Compile the graph (make it executable)
app = workflow.compile()

# STEP 5: Run the graph
print("🚀 Running the graph...\n")
result = app.invoke({"message": "", "counter": 0})

print(f"\n✅ Final State: {result}")