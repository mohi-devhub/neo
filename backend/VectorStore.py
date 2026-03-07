import faiss
import numpy as np
import json
import os
from typing import List, Dict


class VectorStore:
    def __init__(self, index_path: str = "./data/faiss_index.bin", metadata_path: str = "./data/metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.IndexFlatL2
        self.metadata: List[Dict] = []
        self.dimension: int = 0  # inferred from first add_embeddings call or loaded index

        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Loaded FAISS index with {self.index.ntotal} vectors (dim={self.dimension})")
                return
            except Exception as e:
                print(f"Error loading index: {e}. Creating new index.")
        self._create_index()

    def _create_index(self):
        # Dimension is unknown until the first batch of embeddings arrives.
        # Use a sentinel 1-dim index; it will be replaced on the first add_embeddings call.
        self.index = faiss.IndexFlatL2(1)
        self.dimension = 0
        self.metadata = []

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_embeddings(self, embeddings: np.ndarray, documents: List[Dict]) -> None:
        if embeddings.shape[0] != len(documents):
            raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings but {len(documents)} documents")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        incoming_dim = embeddings.shape[1]

        if self.dimension == 0:
            # First batch — initialise index with the actual embedding dimension.
            self.dimension = incoming_dim
            self.index = faiss.IndexFlatL2(self.dimension)
        elif incoming_dim != self.dimension:
            raise ValueError(
                f"Dimension mismatch: index expects {self.dimension}-dim embeddings, "
                f"got {incoming_dim}-dim. Check that the same embedding model is used throughout."
            )

        start_idx = self.index.ntotal
        self.index.add(embeddings)

        for i, doc in enumerate(documents):
            self.metadata.append({'index': start_idx + i, **doc})

        self._save()
        print(f"Added {len(documents)} embeddings. Total vectors: {self.index.ntotal}")

    def has_source(self, filename: str) -> bool:
        """Return True if any vector in the store was ingested from *filename*."""
        return any(m.get("source") == filename for m in self.metadata)

    def remove_source(self, filename: str) -> int:
        """
        Remove all vectors associated with *filename* from the text index.
        Returns the number of vectors removed.
        """
        keep = [m for m in self.metadata if m.get("source") != filename]
        removed = len(self.metadata) - len(keep)
        if removed == 0:
            return 0

        keep_indices = [m["index"] for m in keep]
        rebuild_dim = self.dimension if self.dimension > 0 else self.index.d
        new_index = faiss.IndexFlatL2(rebuild_dim)
        if keep_indices and self.index.ntotal > 0:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            new_vectors = all_vectors[keep_indices]
            new_index.add(new_vectors.astype(np.float32))

        for i, m in enumerate(keep):
            m["index"] = i

        self.index = new_index
        self.dimension = new_index.d
        self.metadata = keep
        self._save()
        print(f"Removed {removed} vector(s) for source '{filename}'. Total: {self.index.ntotal}")
        return removed

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(dist)
                result['score'] = 1.0 / (1.0 + float(dist))
                results.append(result)

        return results
