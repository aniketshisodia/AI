# simple_rag.py

"""
============================================================
🤖 SIMPLE RAG CHATBOT (LangGraph + Ollama + Gemma)
============================================================

📌 PROJECT OVERVIEW
This project implements a simple RAG (Retrieval-Augmented Generation)
chatbot using LangGraph and local LLMs via Ollama.

Instead of relying on general knowledge, the chatbot answers questions
ONLY from a fixed set of documents (doc1.txt → doc4.txt).

------------------------------------------------------------
🧠 HOW THE SYSTEM WORKS

The chatbot follows a pipeline of 4 steps:

1️⃣ LOAD DOCUMENTS
   - Reads text files from disk
   - Splits them into smaller chunks (~300 characters)
   - Converts each chunk into embeddings (vector representation)

2️⃣ RETRIEVE (SEARCH)
   - Converts the user question into an embedding
   - Compares it with all document embeddings
   - Uses cosine similarity to find most relevant chunks
   - Returns top matching chunks (top_k = 2)

3️⃣ GENERATE ANSWER
   - Sends retrieved chunks as context to the LLM (Gemma)
   - LLM generates answer strictly based on that context
   - Prevents hallucination (no external knowledge used)

4️⃣ DISPLAY RESULT
   - Shows final answer
   - Optionally shows sources and similarity scores

------------------------------------------------------------
🔄 LANGGRAPH WORKFLOW

This project uses LangGraph to define a structured pipeline:

    load → search → generate → show → END

Each step is a "node" that performs a single task.
Data flows through a shared State object.

------------------------------------------------------------
📦 STATE OBJECT

The State dictionary carries data across nodes:

    question  → user input
    chunks    → all document chunks + embeddings
    relevant  → top matched chunks
    answer    → final generated response

------------------------------------------------------------
📐 CORE CONCEPTS USED

• Embeddings:
  Convert text into numerical vectors for semantic comparison

• Cosine Similarity:
  Measures similarity between vectors (higher = more relevant)

• Chunking:
  Splits large text into smaller parts for better retrieval

------------------------------------------------------------
⚙️ MODELS USED

• Embedding Model:
  nomic-embed-text

• Language Model:
  gemma3 (fallback: gemma:2b)

------------------------------------------------------------
⚠️ LIMITATIONS

✔ Works only on provided documents
✔ No internet or external knowledge
✔ Reloads documents on every query (not optimized)

------------------------------------------------------------
🎯 PURPOSE OF THIS PROJECT

This is a foundational implementation of RAG, useful for:
- Building ChatGPT-like systems with custom data
- Document-based Q&A systems
- AI assistants for private knowledge bases

------------------------------------------------------------
💡 SUMMARY

This project combines:
    🔍 Search (Retrieval)
    🤖 LLM (Generation)

To create a grounded, reliable chatbot.

============================================================
"""



"""

SIMPLE RAG CHATBOT with LangGraph and Gemma 3
Just 4 text files in the same folder
"""

import os
import numpy as np
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import ollama

# ============================================
# STEP 1: Define what information flows through our graph
# ============================================
class State(TypedDict):
    question: str           # User's question
    chunks: List[Dict]      # Document chunks with embeddings
    relevant: List[Dict]    # Retrieved relevant chunks
    answer: str             # Final answer

# ============================================
# STEP 2: Document Processor - Handles files and searching
# ============================================
class DocumentProcessor:
    def __init__(self):
        self.chunks = []  # Store all document chunks
    
    def load_files(self, file_names):
        """Load text files and convert to embeddings"""
        print("\n📚 Loading your documents...")
        print("-" * 40)
        
        for file_name in file_names:
            # Check if file exists
            if not os.path.exists(file_name):
                print(f"❌ Missing: {file_name}")
                continue
            
            # Read the file
            with open(file_name, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"✅ Loaded: {file_name} ({len(text)} chars)")
            
            # Split into chunks (simple method)
            chunks = self._split_text(text)
            print(f"   → Split into {len(chunks)} chunks")
            
            # Create embedding for each chunk
            for i, chunk in enumerate(chunks):
                embedding = self._get_embedding(chunk)
                self.chunks.append({
                    'text': chunk,
                    'embedding': embedding,
                    'source': file_name,
                    'chunk_id': i
                })
            
            print(f"   ✅ Embedded {len(chunks)} chunks")
        
        print(f"\n📊 Total: {len(self.chunks)} chunks ready\n")
        return self.chunks
    
    def _split_text(self, text):
        """Simple text splitting"""
        # Clean the text
        text = text.replace('\n', ' ')
        
        # Split into chunks of ~300 characters
        chunk_size = 300
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _get_embedding(self, text):
        """Get embedding using Ollama"""
        try:
            response = ollama.embed(model='nomic-embed-text', input=text)
            return response['embeddings'][0]
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            return [0.0] * 768  # Return zeros if error
    
    def search(self, query, top_k=2):
        """Find most relevant chunks for a question"""
        # Get embedding for the question
        query_embedding = self._get_embedding(query)
        
        # Calculate similarity with all chunks
        results = []
        for chunk in self.chunks:
            similarity = self._cosine_similarity(query_embedding, chunk['embedding'])
            results.append({
                'text': chunk['text'],
                'similarity': similarity,
                'source': chunk['source']
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return float(dot / (norm1 * norm2))

# ============================================
# STEP 3: LangGraph Nodes (each does one specific task)
# ============================================

def load_documents(state: State) -> State:
    """NODE 1: Load all documents"""
    print("\n🔧 [1/4] Loading documents...")
    
    processor = DocumentProcessor()
    files = ['doc1.txt', 'doc2.txt', 'doc3.txt', 'doc4.txt']
    chunks = processor.load_files(files)
    
    return {
        'question': state['question'],
        'chunks': chunks,
        'relevant': [],
        'answer': ''
    }

def search_documents(state: State) -> State:
    """NODE 2: Find relevant documents"""
    print("\n🔧 [2/4] Searching for relevant information...")
    print(f"   Question: {state['question']}")
    
    processor = DocumentProcessor()
    processor.chunks = state['chunks']
    
    # Search for relevant chunks
    relevant = processor.search(state['question'], top_k=2)
    
    print(f"\n   Found {len(relevant)} relevant chunks:")
    for i, doc in enumerate(relevant, 1):
        print(f"   {i}. From {doc['source']} (score: {doc['similarity']:.3f})")
        print(f"      {doc['text'][:80]}...")
    
    return {
        'question': state['question'],
        'chunks': state['chunks'],
        'relevant': relevant,
        'answer': ''
    }

def generate_answer(state: State) -> State:
    """NODE 3: Generate answer using Gemma 3"""
    print("\n🔧 [3/4] Generating answer with Gemma 3...")
    
    if not state['relevant']:
        answer = "No relevant information found in the documents."
    else:
        # Build context from relevant documents
        context = ""
        for doc in state['relevant']:
            context += f"[Source: {doc['source']}]\n{doc['text']}\n\n"
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context.

CONTEXT:
{context}

QUESTION: {state['question']}

ANSWER (if the answer is not in the context, say "I cannot answer this based on the documents"):"""
        
        try:
            # Try Gemma 3 first
            response = ollama.chat(
                model='gemma3:latest',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'num_predict': 200}
            )
            answer = response['message']['content']
            print("   ✅ Used Gemma 3")
            
        except:
            # Fallback to Gemma 2B
            print("   ⚠️  Gemma 3 not found, using Gemma 2B")
            response = ollama.chat(
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'num_predict': 200}
            )
            answer = response['message']['content']
    
    return {
        'question': state['question'],
        'chunks': state['chunks'],
        'relevant': state['relevant'],
        'answer': answer
    }

def show_results(state: State) -> State:
    """NODE 4: Display the answer"""
    print("\n🔧 [4/4] Finalizing...")
    return state

# ============================================
# STEP 4: Build the LangGraph workflow
# ============================================

def create_chatbot():
    """Create the graph workflow"""
    
    # Create empty graph
    workflow = StateGraph(State)
    
    # Add nodes (each step)
    workflow.add_node("load", load_documents)
    workflow.add_node("search", search_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("show", show_results)
    
    # Define the flow (order of execution)
    workflow.set_entry_point("load")      # Start here
    workflow.add_edge("load", "search")   # After load, go to search
    workflow.add_edge("search", "generate") # After search, go to generate
    workflow.add_edge("generate", "show") # After generate, go to show
    workflow.add_edge("show", END)        # After show, finish
    
    # Compile the graph
    return workflow.compile()

# ============================================
# STEP 5: Main Chatbot Loop
# ============================================

def print_banner():
    """Welcome message"""
    print("\n" + "="*60)
    print("🤖 SIMPLE RAG CHATBOT")
    print("="*60)
    print("\n📁 Documents loaded:")
    print("   • doc1.txt - Artificial Intelligence")
    print("   • doc2.txt - Climate Change")
    print("   • doc3.txt - Healthcare Technology")
    print("   • doc4.txt - Digital Business")
    print("\n💡 I can only answer questions from these documents")
    print("❓ Type 'quit' to exit")
    print("🔍 Type 'sources' to see where answers came from")
    print("="*60 + "\n")

def main():
    """Run the chatbot"""
    
    print_banner()
    
    # Check if files exist
    required_files = ['doc1.txt', 'doc2.txt', 'doc3.txt', 'doc4.txt']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("❌ Missing files detected!")
        print("Please create these files in the same folder:")
        for f in missing:
            print(f"   • {f}")
        print("\nFile contents are provided in the instructions above.")
        return
    
    # Create the chatbot
    print("🔄 Building chatbot workflow...")
    chatbot = create_chatbot()
    print("✅ Chatbot ready!\n")
    
    # Store last result for sources
    last_result = None
    
    # Interactive loop
    while True:
        # Get question
        question = input("👤 You: ").strip()
        
        # Check exit
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Show sources command
        if question.lower() == 'sources':
            if last_result and last_result['relevant']:
                print("\n📚 Sources for last answer:")
                print("-" * 40)
                for i, doc in enumerate(last_result['relevant'], 1):
                    print(f"\n{i}. File: {doc['source']}")
                    print(f"   Relevance: {doc['similarity']:.3f}")
                    print(f"   Text: {doc['text'][:150]}...")
                print()
            else:
                print("\n⚠️  No sources available. Ask a question first.\n")
            continue
        
        # Skip empty questions
        if not question:
            continue
        
        # Process question
        print("\n" + "🔄" * 30)
        print("PROCESSING YOUR QUESTION")
        print("🔄" * 30)
        
        try:
            # Run the graph
            result = chatbot.invoke({
                'question': question,
                'chunks': [],
                'relevant': [],
                'answer': ''
            })
            
            # Store for sources
            last_result = result
            
            # Display answer
            print("\n" + "="*60)
            print("🤖 ANSWER:")
            print("="*60)
            print(f"\n{result['answer']}\n")
            
            # Display stats
            print("-" * 60)
            print(f"📊 Found {len(result['relevant'])} relevant documents")
            if result['relevant']:
                best_score = result['relevant'][0]['similarity']
                print(f"📈 Best match score: {best_score:.3f}")
            print("-" * 60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running")
            print("2. Run: ollama pull nomic-embed-text")
            print("3. Run: ollama pull gemma3:latest")
            print("4. Or run: ollama pull gemma:2b (fallback)\n")

# ============================================
# STEP 6: Run the program
# ============================================

if __name__ == "__main__":
    main()