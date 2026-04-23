# 01_list_based_rag.py

"""
SIMPLE RAG SYSTEM (NO DATABASE)

What this does:
- Stores documents in a Python list
- Converts them into embeddings using Ollama
- Searches similar documents using cosine similarity
- Sends relevant docs to LLM to generate answer

This is the simplest form of RAG before using vector databases like Qdrant.
"""

# Import required libraries
from typing import TypedDict, List, Dict   # For defining structured state
from langgraph.graph import StateGraph, END  # LangGraph for workflow
import ollama   # For embeddings + LLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # For similarity calculation


# Define the structure of our shared state
class State(TypedDict):
    question: str                     # User question
    documents: List[Dict]             # Stored documents with embeddings
    relevant_docs: List[str]          # Top matching documents
    answer: str                       # Final generated answer


# Simple RAG class (no database, just list)
class SimpleRAG:
    def __init__(self):
        # Store documents as list of dictionaries
        self.documents = []  
        # Each document will look like:
        # {"text": "...", "embedding": [...], "source": "..."}

        
    def add_document(self, text: str, source: str = "unknown"):
        """Add document to memory"""
        
        # Convert text into embedding (vector representation)
        embedding = ollama.embed(
            model='all-minilm', 
            input=text
        )['embeddings'][0]
        
        # Store document along with its embedding
        self.documents.append({
            "text": text,          # Original text
            "embedding": embedding, # Vector form
            "source": source        # Optional source info
        })
        
        # Print confirmation
        print(f"✅ Added: {text[:50]}...")


    def search(self, query: str, top_k: int = 2) -> List[str]:
        """Search for similar documents using cosine similarity"""
        
        # Convert query into embedding
        query_embedding = ollama.embed(
            model='all-minilm', 
            input=query
        )['embeddings'][0]
        
        # Store similarity scores
        similarities = []
        
        # Compare query with each document
        for doc in self.documents:
            
            # Compute cosine similarity
            sim = cosine_similarity(
                [query_embedding],       # Query vector
                [doc["embedding"]]       # Document vector
            )[0][0]
            
            # Store (similarity score, text)
            similarities.append((sim, doc["text"]))
        
        # Sort documents by similarity (highest first)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top_k documents only
        return [text for _, text in similarities[:top_k]]


# ------------------- LANGGRAPH NODES -------------------

def load_documents(state: State) -> State:
    """Load documents into RAG system (runs first)"""
    
    print("📚 Loading documents...")
    
    # Create new RAG instance
    rag = SimpleRAG()
    
    # Sample documents
    docs = [
        ("Python is a high-level programming language known for its simplicity and readability", "doc1"),
        ("LangGraph helps create complex workflows with cycles and conditional logic", "doc2"),
        ("Machine learning is a subset of AI that learns from data without explicit programming", "doc3"),
        ("RAG combines information retrieval with language generation for accurate answers", "doc4"),
        ("Vector embeddings convert text into numbers that capture semantic meaning", "doc5")
    ]
    
    # Add each document into RAG system
    for text, source in docs:
        rag.add_document(text, source)
    
    # Return updated state
    return {
        "question": state["question"],   # Keep user question
        "documents": rag.documents,      # Save documents into state
        "relevant_docs": [],             # Empty initially
        "answer": ""                     # No answer yet
    }


def retrieve_documents(state: State) -> State:
    """Search for relevant documents"""
    
    print(f"\n🔍 Searching for: {state['question']}")
    
    # Create RAG instance and load stored documents
    rag = SimpleRAG()
    rag.documents = state["documents"]
    
    # Perform similarity search
    relevant = rag.search(state["question"], top_k=2)
    
    # Print results
    print(f"   Found {len(relevant)} relevant documents")
    for i, doc in enumerate(relevant, 1):
        print(f"   {i}. {doc[:60]}...")
    
    # Return updated state
    return {
        "question": state["question"],
        "documents": state["documents"],
        "relevant_docs": relevant,   # Store retrieved docs
        "answer": ""
    }


def generate_answer(state: State) -> State:
    """Generate answer using retrieved documents"""
    
    # Combine retrieved docs into context
    context = "\n\n".join(state["relevant_docs"])
    
    # Create prompt for LLM
    prompt = f"""Answer based ONLY on this context:

Context:
{context}

Question: {state["question"]}

Answer:"""
    
    # Call LLM (Ollama)
    response = ollama.chat(
        model='gemma3',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.3}  # Low randomness
    )
    
    # Return final state with answer
    return {
        "question": state["question"],
        "documents": state["documents"],
        "relevant_docs": state["relevant_docs"],
        "answer": response['message']['content']
    }


# ------------------- BUILD GRAPH -------------------

# Create LangGraph workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("load", load_documents)        # Load documents
workflow.add_node("retrieve", retrieve_documents) # Search docs
workflow.add_node("generate", generate_answer)   # Generate answer

# Define flow (edges)
workflow.set_entry_point("load")        # Start from load
workflow.add_edge("load", "retrieve")   # Then retrieve
workflow.add_edge("retrieve", "generate") # Then generate
workflow.add_edge("generate", END)      # End after generation

# Compile graph
app = workflow.compile()


# ------------------- RUN -------------------

print("="*50)
print("SIMPLE LIST-BASED RAG")
print("="*50)

# Run the workflow
result = app.invoke({
    "question": "What is RAG?",   # User query
    "documents": [],              # Empty initially
    "relevant_docs": [],
    "answer": ""
})

# Print final answer
print(f"\n🤖 Answer: {result['answer']}")