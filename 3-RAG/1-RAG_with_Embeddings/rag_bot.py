# rag_bot.py
import ollama
import os
from Embedding_Utils import get_embedding, find_most_similar

print("=" * 60)
print("🤖 RAG CHATBOT WITH EMBEDDINGS")
print("=" * 60)

# ============================================
# STEP 1: LOAD DOCUMENTS FROM 'doc' FOLDER
# ============================================

print("\n📄 Loading documents from 'doc' folder...")

# Path to your documents folder
doc_folder = "docs"

# Check if folder exists
if not os.path.exists(doc_folder):
    print(f"❌ Folder '{doc_folder}' not found!")
    print("   Creating 'doc' folder for you...")
    os.makedirs(doc_folder)
    print("   Please add your .txt files to the 'doc' folder")
    exit(1)

# List all .txt files in the doc folder
files = [f for f in os.listdir(doc_folder) if f.endswith('.txt')]

if not files:
    print(f"❌ No .txt files found in '{doc_folder}' folder!")
    exit(1)

print(f"   Found files: {files}")

all_chunks = []

for file in files:
    file_path = os.path.join(doc_folder, file)  # Full path: doc/cafe_info.txt
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into chunks (simple: split by double newline)
    chunks = content.split('\n\n')
    
    for chunk in chunks:
        if len(chunk.strip()) > 20:  # Ignore empty chunks
            all_chunks.append(chunk.strip())
            print(f"   ✓ Loaded chunk from {file}")

print(f"\n✅ Total chunks: {len(all_chunks)}")

# ============================================
# STEP 2: CREATE EMBEDDINGS FOR ALL CHUNKS
# ============================================

print("\n🔢 Creating embeddings for all chunks...")

chunk_embeddings = []
for i, chunk in enumerate(all_chunks):
    print(f"   Embedding chunk {i+1}/{len(all_chunks)}...", end=" ", flush=True)
    emb = get_embedding(chunk)
    chunk_embeddings.append(emb)
    print("✓")

print("\n" + "=" * 60)
print("💬 CHATBOT READY!")
print("=" * 60)
print("\n💡 Try asking:")
print("   • What drinks do you have?")
print("   • When are you open on Sunday?")
print("   • Do you have WiFi?")
print("   • How much is a latte?")
print("\nType 'quit' to exit")
print("=" * 60)

# ============================================
# STEP 3: CHAT LOOP
# ============================================

while True:
    question = input("\n☕ You: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not question:
        continue
    
    print("\n🔍 Searching for relevant information...")
    
    # Step 1: Embed the question
    question_emb = get_embedding(question)
    
    # Step 2: Find similar chunks
    similar_chunks = find_most_similar(question_emb, chunk_embeddings, all_chunks, top_k=2)
    
    # Step 3: Show what was found
    print(f"📚 Found {len(similar_chunks)} relevant sections:")
    for i, chunk in enumerate(similar_chunks, 1):
        print(f"   {i}. Similarity: {chunk['similarity']:.3f}")
        print(f"      Preview: {chunk['text'][:80]}...")
    
    # Step 4: Build context from similar chunks
    context = "\n\n".join([chunk['text'] for chunk in similar_chunks])
    
    # Step 5: Create prompt with context
    prompt = f"""You are a helpful assistant for Brew Haven Cafe. 
Answer the question using ONLY the information below.

INFORMATION FROM OUR DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the information above
2. Be concise and friendly
3. If the answer isn't in the information, say "I don't have that information"

ANSWER:"""
    
    # Step 6: Generate answer
    print("\n💭 Generating answer...")
    response = ollama.chat(
        model="gemma3",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_predict": 150
        }
    )
    
    # Step 7: Show answer
    print(f"\n🤖 {response['message']['content']}")