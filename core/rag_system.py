# core/rag_system.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Compatibilidad: InMemoryVectorStore cambia según versión
try:
    from langchain_community.vectorstores import InMemoryVectorStore
except Exception:
    try:
        from langchain.vectorstores import InMemoryVectorStore  # type: ignore
    except Exception as e:
        raise ImportError(
            "No se pudo importar InMemoryVectorStore. "
            "Instala/actualiza langchain-community y langchain."
        ) from e


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: Optional[float] = None


class RAGSystem:
    def __init__(
        self,
        documents_path: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        k: int = 4,
    ) -> None:
        self.documents_path = documents_path
        self.k = k

        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)
        self._initialized = False

    def load_markdown_documents(self) -> List[Document]:
        if not os.path.isdir(self.documents_path):
            raise FileNotFoundError(f"No existe la carpeta '{self.documents_path}'.")

        md_files = [f for f in os.listdir(self.documents_path) if f.lower().endswith(".md")]
        if len(md_files) < 2:
            raise ValueError("Debes tener al menos 2 archivos .md en /documents.")

        docs: List[Document] = []
        for filename in md_files:
            path = os.path.join(self.documents_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": filename, "path": path}))
        return docs

    def ingest(self) -> None:
        base_docs = self.load_markdown_documents()
        chunks = self.splitter.split_documents(base_docs)
        self.vectorstore.add_documents(chunks)
        self._initialized = True

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        if not self._initialized:
            raise RuntimeError("Llama a ingest() antes de retrieve().")

        # Con score si existe; si no, sin score.
        try:
            results: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(query, k=self.k)  # type: ignore
            return [
                RetrievedChunk(
                    content=doc.page_content,
                    source=str(doc.metadata.get("source", "unknown")),
                    score=float(score),
                )
                for doc, score in results
            ]
        except Exception:
            docs = self.vectorstore.similarity_search(query, k=self.k)
            return [
                RetrievedChunk(
                    content=doc.page_content,
                    source=str(doc.metadata.get("source", "unknown")),
                    score=None,
                )
                for doc in docs
            ]