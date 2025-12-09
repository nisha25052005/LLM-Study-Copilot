import faiss
import numpy as np
from typing import List, Dict, Tuple


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: List[Dict] = []
        self._is_trained = False

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        embeddings: (n, dim)
        metadata: list of dicts aligned with embeddings.
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        if not self._is_trained and isinstance(self.index, faiss.IndexFlatL2):
            self._is_trained = True

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")

        query_embedding = np.expand_dims(query_embedding, axis=0)  # (1, dim)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx].copy()
            meta["distance"] = float(dist)
            results.append(meta)
        return results
