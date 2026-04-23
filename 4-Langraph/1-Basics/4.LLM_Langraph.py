# 04_llm_graph.py
"""
REAL LLM INTEGRATION: Answer questions with quality checking
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import ollama

class State(TypedDict):
    question: str
    answer: str
    quality: str
    attempts: int

def generate_answer(state: State) -> State:
    """Use LLM to generate answer"""
    question = state["question"]
    attempts = state["attempts"]
     
    print(f"🤔 Attempt {attempts}: Generating answer for: {question}")
    
    # Call Ollama (using tiny model for speed)
    response = ollama.chat(
        model='gemma3',  # or 'tinyllama' if gemma3 is slow
        messages=[{'role': 'user', 'content': f"Answer briefly: {question}"}],
        options={'temperature': 0.3, 'num_predict': 100}
    )
    
    answer = response['message']['content']
    print(f"   Generated: {answer[:100]}...")
    
    return {
        "question": state["question"],
        "answer": answer,
        "quality": "",
        "attempts": state["attempts"]
    }

def check_quality(state: State) -> Literal["good", "bad"]:
    """Check if answer is good quality"""
    answer = state["answer"]
    attempts = state["attempts"]
    
    # Simple quality checks
    if len(answer) < 20:
        print(f"   ❌ Too short ({len(answer)} chars)")
        return "bad"
    elif "?" in answer or "sorry" in answer.lower():
        print(f"   ❌ Contains uncertainty markers")
        return "bad"
    elif attempts >= 3:
        print(f"   ⚠️  Max attempts reached")
        return "good"
    else:
        print(f"   ✅ Quality acceptable")
        return "good"

def improve_answer(state: State) -> State:
    """Try to improve the answer"""
    print(f"🔄 Improving answer...")
    
    # Ask LLM to improve
    response = ollama.chat(
        model='gemma3',
        messages=[
            {'role': 'user', 'content': f"Question: {state['question']}\nMy answer: {state['answer']}\nPlease provide a better, more complete answer:"}
        ],
        options={'temperature': 0.3, 'num_predict': 150}
    )
    
    return {
        "question": state["question"],
        "answer": response['message']['content'],
        "quality": "",
        "attempts": state["attempts"] + 1
    }

# Build graph with loop
workflow = StateGraph(State)

workflow.add_node("generate", generate_answer)
workflow.add_node("improve", improve_answer)

workflow.set_entry_point("generate")

workflow.add_conditional_edges(
    "generate",
    check_quality,
    {
        "good": END,
        "bad": "improve"
    }
)

workflow.add_edge("improve", "generate")

app = workflow.compile()

# Run the graph
print("="*50)
print("ASK A QUESTION")
print("="*50)

question = input("Your question: ")

result = app.invoke({
    "question": question,
    "answer": "",
    "quality": "",
    "attempts": 1
})

print("\n" + "="*50)
print("FINAL ANSWER")
print("="*50)
print(result["answer"])
print(f"\n📊 Attempts needed: {result['attempts']}")