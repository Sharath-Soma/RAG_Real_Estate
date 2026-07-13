from __future__ import annotations

from pathlib import Path
from typing import List

from config import KNOWLEDGE_DIR


class KnowledgeLoader:
    """Loads markdown/text documents from the knowledge base folder."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or KNOWLEDGE_DIR

    def load_documents(self) -> List[dict]:
        documents: List[dict] = []
        if not self.base_dir.exists():
            return documents

        for path in sorted(self.base_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json"}:
                content = path.read_text(encoding="utf-8")
                documents.append({"source": path.name, "content": content})

        return documents
