# ==========================================================
# LESSON 5: VISUALIZING EMBEDDING SIMILARITY (CLUSTERS)
# ==========================================================
#
# This program shows how embeddings group similar words
# together in vector space.
#
# Key Concept:
# Words with similar meaning → high similarity (close vectors)
# Unrelated words → low similarity (far vectors)
#
# Workflow:
# 1. Create embeddings for different categories
# 2. Compare similarity between words
# 3. Observe clustering behavior
#
# Categories:
# - Coffee-related words
# - Food-related words
# - Unrelated objects
# ==========================================================

import ollama
import numpy as np


# ==========================================================
# COSINE SIMILARITY FUNCTION (must be defined first)
# ==========================================================
def cosine_similarity(vec1, vec2):
    """Measure similarity between two embedding vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


print("=" * 60)
print("🎨 VISUALIZING EMBEDDING SIMILARITY")
print("=" * 60)


# ==========================================================
# WORD GROUPS (CATEGORIES)
# ==========================================================

coffee_related = ["coffee", "latte", "espresso", "cappuccino", "mocha"]
food_related = ["croissant", "muffin", "sandwich", "pastry", "bagel"]
unrelated = ["car", "computer", "phone", "book", "table"]

# Combine all words
all_words = coffee_related + food_related + unrelated


# ==========================================================
# STEP 1: CREATE EMBEDDINGS
# ==========================================================

print("\n🔄 Creating embeddings...")

embeddings = {}

for word in all_words:
    response = ollama.embed(model="all-minilm", input=word)

    # Store embedding vector
    embeddings[word] = response['embeddings'][0]

print(f"✅ Created {len(embeddings)} embeddings")


# ==========================================================
# STEP 2: SIMILARITY MATRIX (FIRST 5 WORDS)
# ==========================================================

print("\n📊 Similarity matrix (first 5 words):")

# Print column headers
print("    " + " ".join([f"{w[:8]:8}" for w in all_words[:5]]))

# Compare each word with each other
for word1 in all_words[:5]:
    row = f"{word1[:8]:8}"

    for word2 in all_words[:5]:
        sim = cosine_similarity(embeddings[word1], embeddings[word2])

        row += f"{sim:8.3f}"  # format to 3 decimal places

    print(row)


# ==========================================================
# STEP 3: ANALYZE SIMILARITY RELATIONSHIPS
# ==========================================================

print("\n🔍 INSIGHTS:")

coffee_word = "coffee"

# Compare coffee with food-related words
print("\n☕ Coffee vs Food:")
for food in food_related[:3]:
    sim = cosine_similarity(embeddings[coffee_word], embeddings[food])
    print(f"   '{coffee_word}' vs '{food}': {sim:.3f}")


# Compare coffee with unrelated words
print("\n🚗 Coffee vs Unrelated:")
for unrelated_word in unrelated[:3]:
    sim = cosine_similarity(embeddings[coffee_word], embeddings[unrelated_word])
    print(f"   '{coffee_word}' vs '{unrelated_word}': {sim:.3f}")