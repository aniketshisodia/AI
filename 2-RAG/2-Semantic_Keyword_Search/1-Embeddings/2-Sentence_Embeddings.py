# ==========================================================
# LESSON 5: SENTENCE EMBEDDINGS & SEMANTIC SEARCH
# ==========================================================
#
# This program demonstrates how embeddings can be used to
# match user questions with the most relevant documents,
# even when exact words don't match.
#
# Key Concept:
# Instead of keyword matching, we use MEANING (semantics).
#
# Workflow:
# 1. Convert questions into embeddings (vectors)
# 2. Convert documents into embeddings
# 3. Compare each question with all documents
# 4. Find the most similar document using cosine similarity
#
# This is the foundation of:
# - RAG (Retrieval-Augmented Generation)
# - Semantic search (Google-like understanding)
# - Chatbot knowledge retrieval
# ==========================================================

import ollama
import numpy as np


print("=" * 60)
print("📝 EMBEDDING SENTENCES")
print("=" * 60)


# Different ways users may ask the same thing
questions = [
    "What time do you open?",
    "When does the cafe open?",
    "What are your opening hours?",
    "What's the weather today?",  # Unrelated question
]


# Documents (knowledge base / chunks)
documents = [
    "Brew Haven Cafe opens at 7am on weekdays",
    "We close at 8pm Monday through Friday",
    "Saturday hours are 8am to 6pm",
    "The coffee is made from organic beans",
]


print("\n🔍 Creating embeddings for questions and documents...")


# ==========================================================
# STEP 1: CREATE EMBEDDINGS FOR QUESTIONS
# ==========================================================

question_embeddings = []

for q in questions:
    response = ollama.embed(model="all-minilm", input=q)

    # Store embedding vector for each question
    question_embeddings.append(response['embeddings'][0])

    print(f"   ✓ Question: '{q[:30]}...'")  # preview first 30 chars


# ==========================================================
# STEP 2: CREATE EMBEDDINGS FOR DOCUMENTS
# ==========================================================

doc_embeddings = []

for doc in documents:
    response = ollama.embed(model="all-minilm", input=doc)

    # Store embedding vector for each document
    doc_embeddings.append(response['embeddings'][0])

    print(f"   ✓ Document: '{doc[:30]}...'")


print("\n📊 Finding most relevant document for each question:")
print("-" * 60)


# ==========================================================
# COSINE SIMILARITY FUNCTION
# ==========================================================

def cosine_similarity(vec1, vec2):
    """
    Measures how similar two vectors are.

    Formula:
    similarity = (A · B) / (||A|| * ||B||)

    Returns value between:
    1.0 → identical meaning
    0.0 → unrelated
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)


# ==========================================================
# STEP 3: FIND BEST MATCH FOR EACH QUESTION
# ==========================================================

for i, question in enumerate(questions):

    print(f"\nQuestion: {question}")

    similarities = []

    # Compare this question with ALL documents
    for j, doc_emb in enumerate(doc_embeddings):

        sim = cosine_similarity(question_embeddings[i], doc_emb)

        # Store (document_index, similarity_score)
        similarities.append((j, sim))


    # Sort documents by highest similarity
    similarities.sort(key=lambda x: x[1], reverse=True)


    # Pick best matching document
    best_idx, best_score = similarities[0]

    print(f"   Best match: '{documents[best_idx]}'")
    print(f"   Similarity score: {best_score:.4f}")


    # Simple threshold check
    if best_score > 0.5:
        print("   ✅ Good match! (Semantic understanding works!)")
    else:
        print("   ❌ Poor match")


# ==========================================================
# FINAL INSIGHT
# ==========================================================

print("\n" + "=" * 60)
print("💡 OBSERVATION:")
print("   Even without matching keywords, embeddings find related content!")
print("   'What time do you open?' matches 'opens at 7am'")
print("=" * 60)