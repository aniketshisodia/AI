# embedding_utils.py
import ollama
import numpy as np

def get_embedding(text):
    """Convert text to embedding (list of numbers)"""
    response = ollama.embed(
        model="all-minilm",
        input=text
    )
    return response['embeddings'][0]

def cosine_similarity(vec1, vec2):
    """Measure similarity between two embeddings (1 = identical, 0 = unrelated)"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def find_most_similar(question_emb, chunk_embeddings, chunks, top_k=2):
    """Find the most similar chunks to the question"""
    similarities = []
    
    # Calculate similarity with each chunk
    for chunk_emb in chunk_embeddings:
        sim = cosine_similarity(question_emb, chunk_emb)
        similarities.append(sim)
    
    # Get indices of top-k highest similarities
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return the top chunks with their similarity scores
    results = []
    for idx in top_indices:
        results.append({
            'text': chunks[idx],
            'similarity': similarities[idx]
        })
    
    return results