# 05_rag_graph.py
"""
RAG CHATBOT with LangGraph
Flow: Retrieve → Generate → Check → (if bad) → Retrieve more → Generate again
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# First, set up Qdrant with some sample data (reusing our RAG code)
def setup_vector_db():
    """Create a simple vector database with sample documents"""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Sample documents
    docs = [
        "Python is a programming language great for AI and data science",
        "LangGraph helps build complex AI workflows with cycles and loops",
        "RAG combines search with LLMs for accurate answers",
        "Qdrant is a vector database optimized for similarity search"
    ]
    
    for doc in docs:
        embedding = ollama.embed(model='all-minilm', input=doc)['embeddings'][0]
        client.upsert(
            collection_name="docs",
            points=[PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": doc})]
        )
    
    return client

class State(TypedDict):
    question: str
    retrieved_docs: list
    answer: str
    confidence: int
    attempts: int

def retrieve_documents(state: State) -> State:
    """Search for relevant documents"""
    question = state["question"]
    print(f"🔍 Retrieving docs for: {question}")
    
    # Get embedding for question
    embedding = ollama.embed(model='all-minilm', input=question)['embeddings'][0]
    
    # Search Qdrant
    results = db.search(
        collection_name="docs",
        query_vector=embedding,
        limit=3
    )
    
    docs = [hit.payload["text"] for hit in results]
    print(f"   Found {len(docs)} relevant documents")
    
    return {
        "question": state["question"],
        "retrieved_docs": docs,
        "answer": "",
        "confidence": 0,
        "attempts": state["attempts"] + 1
    }

def generate_answer(state: State) -> State:
    """Generate answer based on retrieved docs"""
    context = "\n".join(state["retrieved_docs"])
    question = state["question"]
    
    prompt = f"""Answer based ONLY on this context:
    
Context: {context}

Question: {question}

Answer:"""
    
    response = ollama.chat(
        model='gemma3',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.3}
    )
    
    print(f"💡 Generated answer")
    return {
        "question": state["question"],
        "retrieved_docs": state["retrieved_docs"],
        "answer": response['message']['content'],
        "confidence": 70,  # Mock confidence
        "attempts": state["attempts"]
    }

def check_confidence(state: State) -> Literal["good", "bad"]:
    """Check if answer is confident enough"""
    if state["confidence"] > 60:
        return "good"
    else:
        return "bad"

def get_more_docs(state: State) -> State:
    """Try to get more relevant documents"""
    print("🔄 Expanding search...")
    # In real implementation, you'd search with different parameters
    return state

# Build the graph
db = setup_vector_db()

workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("expand", get_more_docs)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")

workflow.add_conditional_edges(
    "generate",
    check_confidence,
    {
        "good": END,
        "bad": "expand"
    }
)

workflow.add_edge("expand", "retrieve")  # Loop back

app = workflow.compile()

# Run interactive chat
print("\n" + "="*50)
print("RAG CHATBOT with LangGraph")
print("="*50)

while True:
    question = input("\n❓ You: ").strip()
    if question.lower() == 'quit':
        break
    
    result = app.invoke({
        "question": question,
        "retrieved_docs": [],
        "answer": "",
        "confidence": 0,
        "attempts": 0
    })
    
    print(f"\n🤖 Bot: {result['answer']}")
    print(f"\n📚 Used {len(result['retrieved_docs'])} documents")