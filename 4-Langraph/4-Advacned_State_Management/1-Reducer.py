# 06_advanced_state.py

"""
Advanced State Management with Reducers in LangGraph

👉 WHAT THIS FILE TEACHES:
- How LangGraph updates state
- How to PREVENT overwriting data
- How to ACCUMULATE (append) data using reducers

👉 CORE IDEA:
Normally → state gets REPLACED
With reducers → state gets MERGED (like appending to a list)

👉 KEY TOOL:
Annotated[..., add]
→ tells LangGraph: "use addition instead of replacement"
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from operator import add   # reducer function


# =========================================================
# 🧠 STATE DEFINITION
# =========================================================

class AdvancedState(TypedDict):
    """
    This defines the structure of our state.

    IMPORTANT:
    - messages and history use 'add' reducer → they will APPEND
    - counter has no reducer → it will be REPLACED normally
    """

    # This will ACCUMULATE messages across runs
    messages: Annotated[List[str], add]

    # This will be overwritten every time
    counter: int

    # This will also ACCUMULATE logs/history
    history: Annotated[List[str], add]


# =========================================================
# ⚙️ NODE FUNCTION
# =========================================================

def add_message(state: AdvancedState) -> AdvancedState:
    """
    This node:
    - Reads current state
    - Creates a new message
    - Returns updates (NOT full state)

    IMPORTANT:
    We return only NEW values.
    LangGraph merges them with existing state.
    """

    # Create a new message based on counter
    new_msg = f"Message {state['counter'] + 1}"

    """
    ⚠️ VERY IMPORTANT:

    Even though we return:
        "messages": [new_msg]

    It does NOT overwrite!

    Because of:
        Annotated[..., add]

    LangGraph internally does:
        old_messages + new_messages
    """

    return {
        # This will be APPENDED, not replaced
        "messages": [new_msg],

        # This will be REPLACED normally
        "counter": state["counter"] + 1,

        # This will also be APPENDED
        "history": [f"Step: added {new_msg}"]
    }


# =========================================================
# 🏗️ BUILD GRAPH
# =========================================================

"""
We create a simple graph with only ONE node.

Flow:
START → add_message → END
"""

workflow = StateGraph(AdvancedState)

# Add node
workflow.add_node("add", add_message)

# Set entry point
workflow.set_entry_point("add")

# Define end of flow
workflow.add_edge("add", END)

# Compile into executable app
app = workflow.compile()


# =========================================================
# ▶️ EXECUTION
# =========================================================

"""
We run the graph multiple times.

IMPORTANT:
We pass the RESULT of previous run into next run.

This simulates "persistent memory".
"""


# -------------------------
# 🥇 First Run
# -------------------------
result = app.invoke({
    "messages": [],
    "counter": 0,
    "history": []
})

"""
State after 1st run:

messages = ["Message 1"]
counter = 1
history = ["Step: added Message 1"]
"""

print(f"After 1st: {result['messages']}")


# -------------------------
# 🥈 Second Run
# -------------------------
result = app.invoke(result)

"""
Now LangGraph merges:

OLD messages = ["Message 1"]
NEW messages = ["Message 2"]

Using reducer (add):
→ ["Message 1", "Message 2"]
"""

print(f"After 2nd: {result['messages']}")


# -------------------------
# 🥉 Third Run
# -------------------------
result = app.invoke(result)

"""
Again merge:

["Message 1", "Message 2"] + ["Message 3"]
→ ["Message 1", "Message 2", "Message 3"]
"""

print(f"After 3rd: {result['messages']}")


# =========================================================
# 🎯 FINAL OUTPUT BEHAVIOR
# =========================================================

"""
Without reducers:
→ messages would be overwritten each time

With reducers:
→ messages KEEP GROWING (append behavior)

FINAL OUTPUT:
After 1st: ["Message 1"]
After 2nd: ["Message 1", "Message 2"]
After 3rd: ["Message 1", "Message 2", "Message 3"]


🧠 FINAL MENTAL MODEL:

Each node does NOT mutate state directly.

Instead:
1. Node returns "updates"
2. LangGraph merges them using rules (reducers)

Reducer decides:
→ Replace? (default)
→ Append? (add)
→ Custom logic? (your own function)

This is what makes LangGraph powerful for:
- Chat memory
- Multi-step reasoning
- Agent workflows
"""