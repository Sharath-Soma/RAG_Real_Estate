from __future__ import annotations

from typing import List

import numpy as np
import streamlit as st

from config import EMBEDDING_MODEL


@st.cache_resource(show_spinner=False)
def _load_sentence_transformer(model_name: str):
    """Load and cache the SentenceTransformer model at the process level."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class EmbeddingService:
    """Encapsulates embedding generation with sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        # Delegate to the process-level cached loader – zero extra cost on reruns.
        self.model = _load_sentence_transformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings
