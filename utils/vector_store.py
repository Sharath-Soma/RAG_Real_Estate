from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from config import VECTOR_STORE_DIR


class VectorStore:
    """A thin FAISS wrapper for storing and retrieving embeddings."""

    def __init__(self, index_path: Path | None = None, metadata_path: Path | None = None, manifest_path: Path | None = None) -> None:
        self.index_path = index_path or VECTOR_STORE_DIR / "faiss_index.bin"
        self.metadata_path = metadata_path or VECTOR_STORE_DIR / "metadata.pkl"
        self.manifest_path = manifest_path or VECTOR_STORE_DIR / "manifest.json"
        self.index: faiss.Index | None = None
        self.metadata: List[dict] = []

    def build(self, embeddings: np.ndarray, metadata: List[dict], source_signature: str) -> None:
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))
        self.metadata = metadata
        self._save(source_signature)

    def load(self, expected_signature: str | None = None) -> bool:
        try:
            if not self.index_path.exists() or not self.metadata_path.exists() or not self.manifest_path.exists():
                return False

            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if expected_signature and manifest.get("source_signature") != expected_signature:
                return False

            self.index = faiss.read_index(str(self.index_path))
            with self.metadata_path.open("rb") as handle:
                self.metadata = pickle.load(handle)
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[int, float]]:
        try:
            if self.index is None:
                raise RuntimeError("Vector store is not initialized. Please build or load an index first.")

            scores, indices = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
            results: List[Tuple[int, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((int(idx), float(score)))
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def _save(self, source_signature: str) -> None:
        if self.index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("wb") as handle:
            pickle.dump(self.metadata, handle)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump({"source_signature": source_signature}, handle)

    @staticmethod
    def compute_signature(documents: List[dict]) -> str:
        """Compute a signature of the indexed dataset.

        Production requirement: if document contents change, we must rebuild the index.
        We include a content hash derived from the normalized loader content.
        """
        payload_docs = []
        for doc in documents:
            doc_name = doc.get("document_name")
            doc_path = doc.get("document_path")
            file_type = doc.get("file_type")
            content = doc.get("content", "") or ""
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            payload_docs.append(
                {
                    "document_name": doc_name,
                    "document_path": doc_path,
                    "file_type": file_type,
                    "content_hash": content_hash,
                }
            )
        payload = json.dumps(payload_docs, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

