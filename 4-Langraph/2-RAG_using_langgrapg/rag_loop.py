# simple_rag_with_loop.py

"""
============================================================
🤖 RAG CHATBOT WITH LOOP (Self-Improving Version)
============================================================

This version adds a LOOP that can retry and improve answers!

FLOW WITH LOOP:
    
    START
      ↓
    LOAD DOCUMENTS (once)
      ↓
    SEARCH (find relevant chunks)
      ↓
    CHECK QUALITY ←─────────┐
      ↓                     │
    (if bad) ───────────────┘ (loop back to search)
      ↓
    (if good)
      ↓
    GENERATE ANSWER
      ↓
    END

WHAT THE LOOP DOES:
- Checks if retrieved documents are relevant enough
- If not (score < 0.5), tries to find better documents
- Maximum 3 attempts before giving up
- Ensures better answer quality

============================================================
"""

import os
import numpy as np
from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END
import ollama

# ============================================
# STEP 1: Define State (with loop tracking)
# ============================================
class State(TypedDict):
    question: str           # User's question
    chunks: List[Dict]      # All document chunks
    relevant: List[Dict]    # Retrieved relevant chunks
    answer: str             # Final answer
    attempts: int           # How many search attempts made
    quality_score: float    # How good is the retrieval
    needs_improvement: bool # Should we try again?

# ============================================
# STEP 2: Document Processor
# ============================================
class DocumentProcessor:
    def __init__(self):
        self.chunks = []
    
    def load_files(self, file_names):
        """Load and embed documents"""
        print("\n📚 Loading documents...")
        print("-" * 40)
        
        for file_name in file_names:
            if not os.path.exists(file_name):
                print(f"❌ Missing: {file_name}")
                continue
            
            with open(file_name, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"✅ Loaded: {file_name} ({len(text)} chars)")
            
            chunks = self._split_text(text)
            print(f"   → Split into {len(chunks)} chunks")
            
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
        """Split text into chunks"""
        text = text.replace('\n', ' ')
        chunk_size = 300
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _get_embedding(self, text):
        """Get embedding from Ollama"""
        try:
            response = ollama.embed(model='nomic-embed-text', input=text)
            return response['embeddings'][0]
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            return [0.0] * 768
    
    def search(self, query, top_k=2, exclude_ids=None):
        """Search for relevant chunks (can exclude already used ones)"""
        if exclude_ids is None:
            exclude_ids = []
        
        query_embedding = self._get_embedding(query)
        
        results = []
        for chunk in self.chunks:
            # Skip already used chunks if needed
            if chunk['chunk_id'] in exclude_ids:
                continue
            
            similarity = self._cosine_similarity(query_embedding, chunk['embedding'])
            results.append({
                'text': chunk['text'],
                'similarity': similarity,
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id']
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return float(dot / (norm1 * norm2))

# ============================================
# STEP 3: Global processor (to avoid reloading)
# ============================================
global_processor = DocumentProcessor()
global_loaded = False

# ============================================
# STEP 4: LangGraph Nodes (with loop support)
# ============================================

def load_documents(state: State) -> State:
    """NODE 1: Load documents (only once)"""
    global global_processor, global_loaded
    
    print("\n🔧 [1/5] Loading documents...")
    
    if not global_loaded:
        files = ['doc1.txt', 'doc2.txt', 'doc3.txt', 'doc4.txt']
        global_processor.load_files(files)
        global_loaded = True
    else:
        print("   (Documents already loaded, skipping)")
    
    return {
        'question': state['question'],
        'chunks': global_processor.chunks,
        'relevant': [],
        'answer': '',
        'attempts': 1,
        'quality_score': 0.0,
        'needs_improvement': False
    }

def search_documents(state: State) -> State:
    """NODE 2: Search for relevant documents"""
    print("\n🔧 [2/5] Searching for relevant information...")
    print(f"   Question: {state['question']}")
    print(f"   Attempt: {state['attempts']}/3")
    
    # Get already used chunk IDs
    used_ids = [doc['chunk_id'] for doc in state['relevant']]
    
    # Search (exclude already used chunks on retries)
    exclude = used_ids if state['attempts'] > 1 else []
    relevant = global_processor.search(state['question'], top_k=2, exclude_ids=exclude)
    
    print(f"\n   Found {len(relevant)} relevant chunks:")
    for i, doc in enumerate(relevant, 1):
        print(f"   {i}. From {doc['source']} (score: {doc['similarity']:.3f})")
        print(f"      {doc['text'][:80]}...")
    
    # Calculate quality score (average similarity)
    quality = sum(d['similarity'] for d in relevant) / len(relevant) if relevant else 0
    
    # Combine with previous relevant docs (for multiple attempts)
    all_relevant = state['relevant'] + relevant
    
    return {
        'question': state['question'],
        'chunks': state['chunks'],
        'relevant': all_relevant,
        'answer': '',
        'attempts': state['attempts'],
        'quality_score': quality,
        'needs_improvement': False
    }

def check_quality(state: State) -> State:
    """NODE 3: Check if retrieval quality is good enough"""
    print("\n🔧 [3/5] Checking retrieval quality...")
    
    quality = state['quality_score']
    attempts = state['attempts']
    
    print(f"   Quality score: {quality:.3f}")
    print(f"   Attempts made: {attempts}/3")
    
    # Decision logic
    needs_improvement = False
    
    if quality >= 0.6:
        print("   ✅ Quality is GOOD! Proceeding to answer generation.")
        needs_improvement = False
    elif attempts >= 3:
        print("   ⚠️  Max attempts reached. Proceeding anyway.")
        needs_improvement = False
    else:
        print("   ❌ Quality is POOR. Will search again with different strategy.")
        needs_improvement = True
    
    return {
        'question': state['question'],
        'chunks': state['chunks'],
        'relevant': state['relevant'],
        'answer': '',
        'attempts': attempts,
        'quality_score': quality,
        'needs_improvement': needs_improvement
    }

def improve_search(state: State) -> State:
    """NODE 4: Try to find better documents (the LOOP back node)"""
    print("\n🔧 [4/5] IMPROVING search (retrying with different approach)...")
    
    # Increment attempt counter
    new_attempts = state['attempts'] + 1
    
    print(f"   🔄 Retry #{new_attempts} - looking for different documents")
    
    return {
        'question': state['question'],
        'chunks': state['chunks'],
        'relevant': state['relevant'],  # Keep previous docs
        'answer': '',
        'attempts': new_attempts,
        'quality_score': 0.0,  # Will recalculate
        'needs_improvement': False
    }

def generate_answer(state: State) -> State:
    """NODE 5: Generate final answer"""
    print("\n🔧 [5/5] Generating answer...")
    
    if not state['relevant']:
        answer = "No relevant information found in the documents."
    else:
        # Build context
        context = ""
        for doc in state['relevant']:
            context += f"[Source: {doc['source']} (relevance: {doc['similarity']:.3f})]\n{doc['text']}\n\n"
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Answer based ONLY on the following context.

CONTEXT:
{context}

QUESTION: {state['question']}

ANSWER (if not in context, say "I cannot answer based on the documents"):"""
        
        try:
            response = ollama.chat(
                model='gemma3:latest',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'num_predict': 200}
            )
            answer = response['message']['content']
            print("   ✅ Used Gemma 3")
        except:
            print("   ⚠️  Using Gemma 2B fallback")
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
        'answer': answer,
        'attempts': state['attempts'],
        'quality_score': state['quality_score'],
        'needs_improvement': False
    }

# ============================================
# STEP 5: Routing Functions (for the loop)
# ============================================

def route_after_quality_check(state: State) -> Literal["improve", "generate"]:
    """Decide whether to loop back or continue"""
    if state['needs_improvement']:
        print("\n   🔄 LOOPING BACK to search for better documents...")
        return "improve"
    else:
        print("\n   ➡️  Moving forward to generate answer...")
        return "generate"

def route_after_improve(state: State) -> Literal["search"]:
    """After improvement, always go back to search"""
    return "search"

# ============================================
# STEP 6: Build Graph with Loop
# ============================================

def create_chatbot_with_loop():
    """Create the graph with a loop"""
    
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("load", load_documents)
    workflow.add_node("search", search_documents)
    workflow.add_node("check_quality", check_quality)
    workflow.add_node("improve", improve_search)
    workflow.add_node("generate", generate_answer)
    
    # Define flow with LOOP
    workflow.set_entry_point("load")
    workflow.add_edge("load", "search")
    workflow.add_edge("search", "check_quality")
    
    # CONDITIONAL EDGE - This creates the LOOP!
    workflow.add_conditional_edges(
        "check_quality",
        route_after_quality_check,
        {
            "improve": "improve",  # If needs improvement → go to improve
            "generate": "generate"  # If good → go to generate
        }
    )
    
    # LOOP BACK EDGE
    workflow.add_edge("improve", "search")  # After improve, go back to search
    
    # End edge
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# ============================================
# STEP 7: Main Chatbot
# ============================================

def print_banner():
    print("\n" + "="*60)
    print("🤖 RAG CHATBOT WITH SELF-IMPROVING LOOP")
    print("="*60)
    print("\n📁 Documents loaded:")
    print("   • doc1.txt - Artificial Intelligence")
    print("   • doc2.txt - Climate Change")
    print("   • doc3.txt - Healthcare Technology")
    print("   • doc4.txt - Digital Business")
    print("\n✨ NEW FEATURE: Loop that improves answers!")
    print("   - If retrieval quality < 0.6, it retries")
    print("   - Maximum 3 attempts")
    print("   - Excludes already used documents on retry")
    print("\n💡 I can only answer questions from these documents")
    print("❓ Type 'quit' to exit")
    print("🔍 Type 'sources' to see sources")
    print("📊 Type 'stats' to see loop statistics")
    print("="*60 + "\n")

def main():
    print_banner()
    
    # Check files
    required_files = ['doc1.txt', 'doc2.txt', 'doc3.txt', 'doc4.txt']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("❌ Missing files! Please create them.")
        return
    
    # Create chatbot
    print("🔄 Building chatbot workflow with LOOP...")
    chatbot = create_chatbot_with_loop()
    print("✅ Chatbot ready!\n")
    
    last_result = None
    
    while True:
        question = input("👤 You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if question.lower() == 'sources':
            if last_result and last_result['relevant']:
                print("\n📚 Sources for last answer:")
                for i, doc in enumerate(last_result['relevant'], 1):
                    print(f"\n{i}. File: {doc['source']}")
                    print(f"   Relevance: {doc['similarity']:.3f}")
                    print(f"   Text: {doc['text'][:150]}...")
                print()
            else:
                print("\n⚠️  No sources available.\n")
            continue
        
        if question.lower() == 'stats':
            if last_result:
                print("\n📊 Loop Statistics:")
                print(f"   • Attempts made: {last_result['attempts']}")
                print(f"   • Final quality: {last_result['quality_score']:.3f}")
                print(f"   • Documents used: {len(last_result['relevant'])}")
                if last_result['attempts'] > 1:
                    print(f"   • 🔄 Loop was used! (improved after {last_result['attempts']-1} retries)")
                else:
                    print(f"   • ✅ No loop needed (good quality on first try)")
                print()
            else:
                print("\n⚠️  Ask a question first.\n")
            continue
        
        if not question:
            continue
        
        print("\n" + "🔄" * 35)
        print("PROCESSING WITH LOOP (will retry if needed)")
        print("🔄" * 35)
        
        try:
            result = chatbot.invoke({
                'question': question,
                'chunks': [],
                'relevant': [],
                'answer': '',
                'attempts': 1,
                'quality_score': 0.0,
                'needs_improvement': False
            })
            
            last_result = result
            
            print("\n" + "="*60)
            print("🤖 FINAL ANSWER:")
            print("="*60)
            print(f"\n{result['answer']}\n")
            
            print("-" * 60)
            print(f"📊 Loop Stats:")
            print(f"   • Attempts needed: {result['attempts']}")
            print(f"   • Final quality: {result['quality_score']:.3f}")
            print(f"   • Documents used: {len(result['relevant'])}")
            if result['attempts'] > 1:
                print(f"   • 🔄 Loop activated! (retried {result['attempts']-1} times)")
            print("-" * 60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running")
            print("2. Run: ollama pull nomic-embed-text")
            print("3. Run: ollama pull gemma3:latest\n")

if __name__ == "__main__":
    main()