"""Knowledge base search module for document exploration."""

from typing import List, Dict, Any
from pathlib import Path


class KnowledgeBaseSearch:
    """Search and filter documents in the knowledge base."""

    def __init__(self, documents: List[Dict[str, Any]]):
        """Initialize with loaded documents."""
        self.documents = documents
        self.unique_docs = self._get_unique_documents()

    def _get_unique_documents(self) -> List[Dict[str, Any]]:
        """Get unique documents (by document_name)."""
        seen = set()
        unique = []
        for doc in self.documents:
            doc_name = doc.get("document_name", "")
            if doc_name not in seen:
                seen.add(doc_name)
                unique.append({
                    "name": doc_name,
                    "path": doc.get("document_path", ""),
                    "file_type": doc.get("file_type", ""),
                    "pages": len([d for d in self.documents if d.get("document_name") == doc_name]),
                    "preview": self._get_preview(doc),
                })
        return sorted(unique, key=lambda x: x["name"])

    def _get_preview(self, document: Dict[str, Any]) -> str:
        """Extract preview text from document."""
        content = document.get("content", "")
        if len(content) > 200:
            return content[:197] + "..."
        return content

    def search_by_name(self, query: str) -> List[Dict[str, Any]]:
        """Search documents by name (case-insensitive)."""
        query_lower = query.lower()
        results = []
        for doc in self.unique_docs:
            if query_lower in doc["name"].lower():
                results.append(doc)
        return results

    def search_by_type(self, file_type: str) -> List[Dict[str, Any]]:
        """Search documents by file type."""
        file_type_lower = file_type.lower()
        return [doc for doc in self.unique_docs if doc["file_type"].lower() == file_type_lower]

    def get_all_types(self) -> List[str]:
        """Get all unique file types in knowledge base."""
        types = set()
        for doc in self.unique_docs:
            if doc["file_type"]:
                types.add(doc["file_type"])
        return sorted(list(types))

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "total_documents": len(self.unique_docs),
            "total_chunks": len(self.documents),
            "file_types": self.get_all_types(),
            "avg_pages_per_doc": sum(doc["pages"] for doc in self.unique_docs) / len(self.unique_docs) if self.unique_docs else 0,
            "total_pages": sum(doc["pages"] for doc in self.unique_docs),
        }

    def get_document_by_name(self, name: str) -> Dict[str, Any]:
        """Get full document information by name."""
        for doc in self.unique_docs:
            if doc["name"] == name:
                return doc
        return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        return self.unique_docs
