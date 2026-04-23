import os           #for files
import numpy as np  #for cosine similarity
from typing import TypedDict , List , Dict 
from langgraph.graph import StateGraph, END
import ollama   


class State(TypedDict):
    question: str           # User's question
    chunks: List[Dict]
    relevant: List[Dict]
    answer : str


class DocumentProcessor:
    def __init__(self):
        self.chunks = []
    
    def load_files(self , file_names):
        print("\n📚 Loading documents...")
        print("-"*40)

        for file_name in file_names:
            if not os.path.exists(file_name):
                print(f"❌ Missing: {file_name}")
        
        with open(file_name , 'r' , encoding='utf-8') as f:
            text = f.read()
        
        print(f"✅ Loaded: {file_name} ({len(text)} chars)")

        chunks = self._split_text(text)
        print(f"   → Split into {len(chunks)} chunks")

        for i , chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk)
            self.chunks.append({
                'text': chunk,
                'embedding': embedding,
                'source': file_name,
                'chunk_id': i
            })
        
        print(f"\n📊 Total: {len(self.chunks)} chunks ready\n")
        return self.chunks
    
    def _split_text(self , text):
        text = text.replace('\n' , ' ')
        chunk_size = 300
        chunks = []
        for i in range(0 , len(text) , chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks
    
    def _get_embedding(self , text) :
        try:
            response = ollama.embed(model='nomic-embed-text' , input=text)
            return response['embeddings'][0]
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            return [0.0] * 768
    

    def search(self , query , top_k=2):
        query_embedding = self._get_embedding(query)
        results = []    

        for chunk in self.chunks:
            similarity = self._cosine_similarity(query_embedding , chunk['embedding'])
            results.append({
                'text': chunk['text'],
                'similarity': similarity,
                'source': chunk['source'],
            })

            results.sort(key=lambda x: x['similarity'] , reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self , vec1 , vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot = np.dot(vec1 , vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)            

        if norm1 == 0 or norm2 == 0 :
            return 0

        return float(dot /(norm1 * norm2))

    def load_document(state: State) -> State:
        processor = DocumentProcessor()
        files = ['doc1.txt' , 'doc2.txt' , 'doc3.txt' , 'doc4.txt']
        chunks = processor.load_files(files)

        return {
            'question': state['question'],
            'chunks': chunks,
            'relevant': [],
            'answer': ''
        }