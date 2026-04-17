# lesson4_simple_rag.py
import ollama

print("=" * 60)
print("🤖 LESSON 4: SIMPLE RAG CHATBOT")
print("=" * 60)

# STEP 1: LOAD DOCUMENTS
def load_documents():
    """Load text from files"""
    documents = []
    
    # List of files to load
    files = ["cafe_info.txt"]
    
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "text": content,
                    "source": file
                })
            print(f"✅ Loaded: {file}")
        except FileNotFoundError:
            print(f"❌ File not found: {file}")
            print("   Creating sample file for you...")
            
            # Create sample file if it doesn't exist
            sample_content = """Brew Haven Cafe was founded in 2019 by Sarah Johnson.
The CEO is Michael Chen.
Located at 123 Coffee Lane, Downtown Metropolis.
We serve artisanal coffee, fresh pastries, and healthy sandwiches.
Our signature drinks include Honey Lavender Latte and Caramel Brulée Cold Brew.
Hours: Monday-Friday 7am-8pm, Saturday 8am-6pm, Sunday 9am-5pm."""
            
            with open(file, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            print(f"✅ Created sample: {file}")
            
            # Load it again
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "text": content,
                    "source": file
                })
    
    return documents

# STEP 2: SIMPLE SEARCH (Keyword matching)
def simple_search(question, documents):
    """Find documents containing keywords from question"""
    # Extract keywords from question (simple version)
    keywords = question.lower().split()
    
    # Remove common words (stop words)
    stop_words = ['what', 'is', 'are', 'do', 'you', 'have', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with']
    keywords = [word for word in keywords if word not in stop_words]
    
    print(f"\n🔍 Searching for keywords: {keywords}")
    
    # Search each document
    relevant_chunks = []
    for doc in documents:
        doc_lower = doc["text"].lower()
        
        # Count how many keywords appear in this document
        score = 0
        for keyword in keywords:
            if keyword in doc_lower:
                score += 1
        
        if score > 0:
            relevant_chunks.append({
                "text": doc["text"],
                "source": doc["source"],
                "score": score
            })
    
    # Sort by score (highest first)
    relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    return relevant_chunks

# STEP 3: BUILD RAG PROMPT
def build_rag_prompt(question, context_chunks):
    """Create a prompt with context for the LLM"""
    if not context_chunks:
        return f"""You are a cafe assistant. The user asked: {question}
        
Unfortunately, I don't have any information about this in my documents.

Please say: "I don't have that information in my documents. Please check with our staff!" """

    # Combine all relevant text
    context = "\n\n".join([chunk["text"] for chunk in context_chunks])
    
    prompt = f"""You are a helpful assistant for Brew Haven Cafe. Answer the question using ONLY the information below.

INFORMATION FROM OUR DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the information above
2. If the answer isn't in the information, say "I don't have that information"
3. Be friendly and concise
4. Don't mention that you're using documents

ANSWER:"""
    
    return prompt

# STEP 4: MAIN CHAT LOOP
def main():
    # Load documents
    print("\n📄 Loading documents...")
    documents = load_documents()
    print(f"\n✅ Loaded {len(documents)} document(s)")
    
    print("\n" + "=" * 60)
    print("💬 CHATBOT READY! (Simple RAG Version)")
    print("=" * 60)
    print("\nAsk questions about Brew Haven Cafe:")
    print("  • What's your signature drink?")
    print("  • Where are you located?")
    print("  • What are your hours?")
    print("\nType 'quit' to exit")
    print("=" * 60)
    
    while True:
        # Get question from user
        question = input("\n☕ You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        # STEP 1: Search for relevant information
        print("\n🔍 Searching documents...")
        relevant_chunks = simple_search(question, documents)
        
        # STEP 2: Show what was found (for learning)
        if relevant_chunks:
            print(f"📚 Found {len(relevant_chunks)} relevant section(s):")
            for i, chunk in enumerate(relevant_chunks, 1):
                print(f"   {i}. From {chunk['source']} (score: {chunk['score']})")
                print(f"      Preview: {chunk['text'][:100]}...")
        else:
            print("📚 No relevant information found")
        
        # STEP 3: Build prompt with context
        prompt = build_rag_prompt(question, relevant_chunks)
        
        # STEP 4: Get answer from LLM
        print("\n💭 Generating answer...")
        response = ollama.chat(
            model="gemma3:2b",  # Use smaller model for faster learning
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,  # Low temp for factual answers
                "num_predict": 150   # Limit response length
            }
        )
        
        # Display answer
        answer = response['message']['content']
        print(f"\n🤖 {answer}\n")

# Run the chatbot
if __name__ == "__main__":
    main()