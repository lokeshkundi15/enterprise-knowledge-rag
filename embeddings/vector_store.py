import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

class VectorStoreManager:
    """Manages dense vector indexing and retrieval using ChromaDB and SentenceTransformers."""
    
    def __init__(self, collection_name: str = "enterprise_knowledge"):
        self.collection_name = collection_name
        self.model = SentenceTransformer(MODEL_NAME)
        
        # FIX: Streamlit Cloud లో "readonly database" ఎర్రర్ రాకుండా In-Memory / Ephemeral Client వాడటం
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Generates embeddings and stores chunks with rich metadata into ChromaDB."""
        if not chunks:
            print("⚠️ No chunks to index.")
            return

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]

        print(f"🔄 Generating dense embeddings using {MODEL_NAME}...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

        print(f"📥 Inserting {len(chunks)} chunks into ChromaDB collection '{self.collection_name}'...")
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("✅ Vector indexing completed successfully.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs cosine similarity search for a query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        formatted_results = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })

        return formatted_results