# ==========================================================
# LESSON 5: UNDERSTANDING EMBEDDINGS (SEMANTIC MEANING)
# ==========================================================
#
# This program demonstrates how words are converted into
# numerical vectors (embeddings) and how we can compare
# their meanings using math.
#
# Key Concept:
# Embeddings = converting text → list of numbers
#
# These numbers capture the "meaning" of words such that:
# - Similar words → similar vectors
# - Different words → different vectors
#
# Workflow:
# 1. Generate embedding for a single word
# 2. Generate embeddings for multiple words
# 3. Compare similarity between words using cosine similarity
#
# Why this matters:
# Embeddings are the foundation of:
# - Semantic search (Google-like search)
# - RAG (Retrieval-Augmented Generation)
# - Recommendation systems
# ==========================================================

import ollama
import numpy as np


# Print header
print("=" * 60)
print("🔢 LESSON 5: UNDERSTANDING EMBEDDINGS")
print("=" * 60)


# ==========================================================
# STEP 1: CREATE EMBEDDING FOR A SINGLE WORD
# ==========================================================

word = "coffee"
print(f"\n1️⃣ Creating embedding for: '{word}'")

# Call embedding model
response = ollama.embed(
    model="all-minilm",  # Small and fast embedding model
    input=word           # Input text to convert into vector
)

# Extract embedding vector (list of numbers)
embedding = response['embeddings'][0]  # First (and only) embedding

# Display information about embedding
print(f"   ✅ Created embedding with {len(embedding)} numbers")  # vector size
print(f"   First 10 numbers: {embedding[:10]}")  # preview
print(f"   Type: {type(embedding)}")  # should be list


# ==========================================================
# STEP 2: CREATE EMBEDDINGS FOR MULTIPLE WORDS
# ==========================================================

words = ["coffee", "beverage", "car", "tea", "latte", "truck"]
print(f"\n2️⃣ Creating embeddings for: {words}")

# Dictionary to store word → embedding mapping
embeddings = {}

# Loop through each word and generate embedding
for w in words:
    response = ollama.embed(model="all-minilm", input=w)

    # Store embedding vector
    embeddings[w] = response['embeddings'][0]

    print(f"   ✓ '{w}' → {len(embeddings[w])} numbers")


# ==========================================================
# STEP 3: COMPARE SIMILARITY BETWEEN WORDS
# ==========================================================

print("\n3️⃣ Calculating similarities (cosine similarity):")
print("   (1.0 = identical meaning, 0.0 = no relation)\n")


# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    """
    Cosine similarity measures how similar two vectors are.
    
    Formula:
    similarity = (A · B) / (||A|| * ||B||)
    
    Where:
    - A · B = dot product
    - ||A|| = magnitude of vector A
    """
    
    dot_product = np.dot(vec1, vec2)      # Multiply and sum
    norm1 = np.linalg.norm(vec1)          # Length of vector 1
    norm2 = np.linalg.norm(vec2)          # Length of vector 2
    
    return dot_product / (norm1 * norm2)


# Compare "coffee" with other words
coffee_vec = embeddings["coffee"]

for word in ["beverage", "tea", "latte", "car", "truck"]:
    sim = cosine_similarity(coffee_vec, embeddings[word])
    
    print(f"   'coffee' vs '{word}': {sim:.4f}")


# ==========================================================
# FINAL INSIGHT
# ==========================================================

print("\n" + "=" * 60)
print("💡 KEY INSIGHT:")
print("   Words with similar meanings have HIGHER similarity scores!")
print("   'coffee' vs 'beverage' > 'coffee' vs 'car'")
print("=" * 60)