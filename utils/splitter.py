from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveTextSplitter:
    """Wraps LangChain's recursive splitter with the assessment's chunking settings."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 150) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return self.splitter.split_text(text)
