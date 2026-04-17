# rag_bot_with_sessions.py
import ollama
import os
from embedding_utils import get_embedding, find_most_similar
from session_manager import SessionManager

print("=" * 60)
print("🤖 RAG CHATBOT WITH SESSION MEMORY")
print("=" * 60)

# ============================================
# STEP 1: LOAD DOCUMENTS
# ============================================

print("\n📄 Loading documents from 'doc' folder...")

doc_folder = "doc"
if not os.path.exists(doc_folder):
    print(f"❌ Folder '{doc_folder}' not found!")
    exit(1)

files = [f for f in os.listdir(doc_folder) if f.endswith('.txt')]

if not files:
    print(f"❌ No .txt files found in '{doc_folder}' folder!")
    exit(1)

print(f"   Found files: {files}")

all_chunks = []

for file in files:
    file_path = os.path.join(doc_folder, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = content.split('\n\n')
    for chunk in chunks:
        if len(chunk.strip()) > 20:
            all_chunks.append(chunk.strip())
            print(f"   ✓ Loaded chunk from {file}")

print(f"\n✅ Total chunks: {len(all_chunks)}")

# ============================================
# STEP 2: CREATE EMBEDDINGS
# ============================================

print("\n🔢 Creating embeddings for all chunks...")

chunk_embeddings = []
for i, chunk in enumerate(all_chunks):
    print(f"   Embedding chunk {i+1}/{len(all_chunks)}...", end=" ", flush=True)
    emb = get_embedding(chunk)
    chunk_embeddings.append(emb)
    print("✓")

# ============================================
# STEP 3: SESSION MANAGEMENT
# ============================================

session_mgr = SessionManager()

print("\n" + "=" * 60)
print("📋 SESSION MENU")
print("=" * 60)
print("1. Start new session")
print("2. Load existing session")
print("3. List all sessions")

choice = input("\nChoose (1-3): ").strip()

if choice == "2":
    sessions = session_mgr.list_sessions()
    if sessions:
        print("\nAvailable sessions:")
        for i, s in enumerate(sessions, 1):
            print(f"   {i}. {s['id']} - {s['created']} ({s['messages']} messages)")
        
        try:
            idx = int(input("\nSelect session number: ")) - 1
            if 0 <= idx < len(sessions):
                session_mgr.load_session(sessions[idx]['id'])
            else:
                print("Invalid. Starting new session...")
                session_mgr.new_session()
        except:
            print("Invalid. Starting new session...")
            session_mgr.new_session()
    else:
        print("No sessions found. Starting new...")
        session_mgr.new_session()

elif choice == "3":
    sessions = session_mgr.list_sessions()
    if sessions:
        print("\nAll sessions:")
        for s in sessions:
            print(f"   • {s['id']} - {s['created']} ({s['messages']} messages)")
    session_mgr.new_session()

else:
    session_mgr.new_session()

print("\n" + "=" * 60)
print(f"💬 ACTIVE SESSION: {session_mgr.current_session_id}")
print("=" * 60)
print("\n💡 Try asking:")
print("   • What drinks do you have?")
print("   • My name is Aniket")
print("   • What's my name? (Tests memory!)")
print("\n🔧 Commands:")
print("   • /new - Start new session")
print("   • /save - Save current session")
print("   • /list - List all sessions")
print("   • /load <id> - Load a session")
print("   • /history - Show conversation history")
print("   • /quit - Exit")
print("=" * 60)

# ============================================
# STEP 4: CHAT LOOP
# ============================================

while True:
    try:
        question = input(f"\n☕ [{session_mgr.current_session_id}] You: ").strip()
        
        if question.lower() in ['quit', '/quit', 'exit', 'q']:
            print(f"\n👋 Session {session_mgr.current_session_id} saved!")
            print("Thanks for visiting Brew Haven Cafe!")
            break
        
        if not question:
            continue
        
        # Handle commands
        if question == '/new':
            session_mgr.new_session()
            continue
        elif question == '/save':
            session_mgr._save()
            print("✅ Session saved")
            continue
        elif question == '/list':
            sessions = session_mgr.list_sessions()
            if sessions:
                print("\nAll sessions:")
                for s in sessions:
                    print(f"   • {s['id']} - {s['created']} ({s['messages']} messages)")
            else:
                print("No sessions found")
            continue
        elif question == '/history':
            if session_mgr.current_history:
                print("\n📜 Conversation History:")
                for i, ex in enumerate(session_mgr.current_history, 1):
                    print(f"\n{i}. [{ex['timestamp']}]")
                    print(f"   You: {ex['user']}")
                    print(f"   Bot: {ex['bot'][:100]}...")
            else:
                print("No history yet")
            continue
        elif question.startswith('/load'):
            parts = question.split()
            if len(parts) > 1:
                session_mgr.load_session(parts[1])
            else:
                print("Usage: /load <session_id>")
            continue
        
        print("\n🔍 Searching for relevant information...")
        
        # Get conversation context (for memory)
        conv_context = session_mgr.get_recent_context()
        
        # Embed the question
        question_emb = get_embedding(question)
        
        # Find similar document chunks
        similar_chunks = find_most_similar(question_emb, chunk_embeddings, all_chunks, top_k=2)
        
        # Build document context
        doc_context = "\n\n".join([chunk['text'] for chunk in similar_chunks])
        
        # Combine conversation memory + document context
        if conv_context and doc_context:
            full_context = conv_context + "\n\n" + doc_context
        elif conv_context:
            full_context = conv_context
        else:
            full_context = doc_context
        
        # Show what was found
        print(f"📚 Found {len(similar_chunks)} relevant sections:")
        for i, chunk in enumerate(similar_chunks, 1):
            print(f"   {i}. Similarity: {chunk['similarity']:.3f}")
        
        # Create prompt with both contexts
        prompt = f"""You are a helpful assistant for Brew Haven Cafe.

{full_context}

QUESTION: {question}

INSTRUCTIONS:
1. Use the conversation history to remember what was discussed
2. Use the document information for facts about the cafe
3. If someone tells you their name, remember it for future questions
4. Be friendly and concise

ANSWER:"""
        
        print("\n💭 Generating answer...")
        response = ollama.chat(
            model="gemma3",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 150
            }
        )
        
        answer = response['message']['content']
        print(f"\n🤖 {answer}")
        
        # Save to session history
        session_mgr.add_exchange(question, answer)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        continue