import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional

class VectorStore:
    def __init__(self, index_path: str = "./data/faiss_index.bin", metadata_path: str = "./data/metadata.json"):
        """
        Initialize FAISS vector store.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata (document info, chunk text, etc.)
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: List[Dict] = []
        self.dimension = 768  # Default embedding dimension (nomic-embed-text, bge-m3)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        
        # Load existing index and metadata if available
        self.load()
    
    def load(self):
        """Load FAISS index and metadata from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Loaded FAISS index with {self.index.ntotal} vectors from {self.index_path}")
            except Exception as e:
                print(f"Error loading index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        print(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, documents: List[Dict]) -> None:
        """
        Add embeddings and metadata to the index.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            documents: List of dicts with keys: 'text', 'source', 'doc_id', 'chunk_id'
        """
        if embeddings.shape[0] != len(documents):
            raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings but {len(documents)} documents")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Add metadata
        for i, doc in enumerate(documents):
            self.metadata.append({
                'index': self.index.ntotal - len(documents) + i,
                **doc
            })
        
        self.save()
        print(f"Added {len(documents)} embeddings. Total vectors: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: numpy array of shape (embedding_dim,) or (1, embedding_dim)
            k: Number of results to return
        
        Returns:
            List of dicts with 'text', 'source', 'distance', 'score'
        """
        if self.index.ntotal == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(dist)
                result['score'] = 1.0 / (1.0 + float(dist))  # Convert distance to similarity score
                results.append(result)
        
        return results
    
    def delete_by_document(self, doc_id: str) -> int:
        """
        Delete all chunks of a document by doc_id.
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            Number of chunks deleted
        """
        indices_to_keep = [i for i, m in enumerate(self.metadata) if m.get('doc_id') != doc_id]
        
        if len(indices_to_keep) == len(self.metadata):
            return 0  # No documents deleted
        
        # Rebuild index without deleted documents
        deleted_count = len(self.metadata) - len(indices_to_keep)
        
        if len(indices_to_keep) == 0:
            # All documents deleted, reset
            self._create_new_index()
            self.metadata = []
        else:
            # Rebuild index with remaining vectors
            kept_metadata = [self.metadata[i] for i in indices_to_keep]
            
            # Extract embeddings from existing index (re-add them)
            # For now, we'll mark as deleted and rebuild on next add
            self.metadata = kept_metadata
        
        self.save()
        print(f"Deleted {deleted_count} chunks from document {doc_id}")
        return deleted_count
    
    def get_document_info(self, doc_id: str) -> Dict:
        """Get metadata for a specific document."""
        chunks = [m for m in self.metadata if m.get('doc_id') == doc_id]
        if not chunks:
            return {}
        
        return {
            'doc_id': doc_id,
            'source': chunks[0].get('source'),
            'chunk_count': len(chunks),
            'created_at': chunks[0].get('created_at')
        }
    
    def list_documents(self) -> List[Dict]:
        """List all unique documents in the index."""
        seen_docs = {}
        for m in self.metadata:
            doc_id = m.get('doc_id')
            if doc_id and doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    'doc_id': doc_id,
                    'source': m.get('source'),
                    'chunk_count': sum(1 for x in self.metadata if x.get('doc_id') == doc_id),
                    'created_at': m.get('created_at')
                }
        return list(seen_docs.values())
    
    def save(self) -> None:
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'total_documents': len(set(m.get('doc_id') for m in self.metadata)),
            'embedding_dimension': self.dimension,
            'metadata_entries': len(self.metadata)
        }
