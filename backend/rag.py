from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import uuid
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import httpx
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader


class RAGError(Exception):
    pass


class RAGStore:
    def __init__(
        self,
        persist_path: str | Path = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        max_upload_bytes: int = 15 * 1024 * 1024,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_upload_bytes = max_upload_bytes

        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        client = chromadb.PersistentClient(path=str(persist_path))

        self.chunks = client.get_or_create_collection(
            name="knowledge_chunks",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.sources = client.get_or_create_collection(
            name="knowledge_sources",
        )

    @staticmethod
    def _source_id(url: str) -> str:
        return "blog-" + hashlib.sha256(url.encode("utf-8")).hexdigest()

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _split_text(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                boundary = max(
                    text.rfind(". ", start, end),
                    text.rfind(" ", start, end),
                )
                if boundary > start + self.chunk_size // 2:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _replace_chunks(
        self,
        source_id: str,
        text: str,
        *,
        source_type: str,
        source_name: str,
        source_url: str = "",
    ) -> int:
        chunks = self._split_text(text)

        if not chunks:
            raise RAGError("Não foi possível extrair texto útil da fonte.")

        self.chunks.delete(where={"source_id": source_id})

        ids = [f"{source_id}-{index}" for index in range(len(chunks))]
        metadata = [
            {
                "source_id": source_id,
                "source_type": source_type,
                "source_name": source_name[:500],
                "source_url": source_url[:2000],
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        for start in range(0, len(chunks), 100):
            end = start + 100
            self.chunks.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                metadatas=metadata[start:end],
            )

        return len(chunks)

    def add_document(
        self,
        filename: str,
        content: bytes,
        content_type: str | None,
    ) -> dict[str, object]:
        if not content:
            raise RAGError("O arquivo está vazio.")

        if len(content) > self.max_upload_bytes:
            raise RAGError(
                f"O arquivo excede o limite de {self.max_upload_bytes // 1024 // 1024} MB."
            )

        extension = Path(filename).suffix.lower()

        if extension == ".pdf" or content_type == "application/pdf":
            try:
                reader = PdfReader(BytesIO(content))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:
                raise RAGError("Não foi possível ler o PDF.") from exc
        elif extension in {".txt", ".md", ".csv"} or (
            content_type and content_type.startswith("text/")
        ):
            text = content.decode("utf-8", errors="replace")
        else:
            raise RAGError("Formato não suportado. Envie PDF, TXT, MD ou CSV.")

        source_id = f"file-{uuid.uuid4()}"
        chunk_count = self._replace_chunks(
            source_id,
            text,
            source_type="document",
            source_name=filename,
        )

        self.sources.upsert(
            ids=[source_id],
            documents=[filename],
            metadatas=[
                {
                    "kind": "document",
                    "name": filename[:500],
                    "fingerprint": self._fingerprint(text),
                }
            ],
        )

        return {
            "source_id": source_id,
            "filename": filename,
            "chunks": chunk_count,
        }

    @staticmethod
    def _validate_public_url(url: str) -> str:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise RAGError("A URL deve usar http ou https.")

        try:
            addresses = socket.getaddrinfo(parsed.hostname, None)
            for address in addresses:
                ip = ipaddress.ip_address(address[4][0])
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                ):
                    raise RAGError("URLs locais ou privadas não são permitidas.")
        except socket.gaierror as exc:
            raise RAGError("Não foi possível resolver o domínio informado.") from exc

        return url

    def _scrape_blog(self, url: str) -> tuple[str, str]:
        safe_url = self._validate_public_url(url)

        try:
            response = httpx.get(
                safe_url,
                timeout=20,
                follow_redirects=True,
                headers={"User-Agent": "Spirit-RAG/1.0"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RAGError("Não foi possível acessar o blog informado.") from exc

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            raise RAGError("A URL não retornou uma página HTML.")

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else safe_url
        main = soup.select_one("article, main") or soup.body

        if not main:
            raise RAGError("Não foi encontrado conteúdo no blog.")

        text = main.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 100:
            raise RAGError("O conteúdo do blog é insuficiente para indexação.")

        return title, text

    def add_blog(self, url: str, *, force: bool = False) -> dict[str, object]:
        url = self._validate_public_url(url)
        source_id = self._source_id(url)

        existing = self.sources.get(ids=[source_id], include=["metadatas"])
        if existing["ids"] and not force:
            return {
                "status": "already_indexed",
                "url": url,
                "message": "Este blog já está cadastrado.",
            }

        title, text = self._scrape_blog(url)
        fingerprint = self._fingerprint(text)

        if existing["ids"]:
            old_fingerprint = existing["metadatas"][0].get("fingerprint")
            if old_fingerprint == fingerprint:
                return {
                    "status": "unchanged",
                    "url": url,
                    "message": "Nenhum conteúdo novo foi encontrado.",
                }

        chunk_count = self._replace_chunks(
            source_id,
            text,
            source_type="blog",
            source_name=title,
            source_url=url,
        )

        self.sources.upsert(
            ids=[source_id],
            documents=[title],
            metadatas=[
                {
                    "kind": "blog",
                    "url": url,
                    "title": title[:500],
                    "fingerprint": fingerprint,
                }
            ],
        )

        return {
            "status": "indexed",
            "url": url,
            "title": title,
            "chunks": chunk_count,
        }

    def refresh_blogs(self) -> list[dict[str, object]]:
        result = self.sources.get(
            where={"kind": "blog"},
            include=["metadatas"],
        )

        refreshed: list[dict[str, object]] = []

        for metadata in result["metadatas"]:
            url = metadata.get("url")
            if not url:
                continue

            try:
                refreshed.append(self.add_blog(url, force=True))
            except RAGError as exc:
                refreshed.append({
                    "status": "error",
                    "url": url,
                    "message": str(exc),
                })

        return refreshed

    def search(self, query: str, limit: int = 4) -> list[dict[str, object]]:
        if not query.strip() or self.chunks.count() == 0:
            return []

        result = self.chunks.query(
            query_texts=[query],
            n_results=min(limit, self.chunks.count()),
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "text": document,
                "source_name": metadata.get("source_name", ""),
                "source_url": metadata.get("source_url", ""),
                "distance": distance,
            }
            for document, metadata, distance in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def context_for(self, query: str, limit: int = 4) -> str:
        matches = self.search(query, limit)

        if not matches:
            return ""

        parts = []

        for index, item in enumerate(matches, start=1):
            source = item["source_url"] or item["source_name"] or "Documento local"
            parts.append(
                f"[Fonte {index}: {source}]\n{item['text']}"
            )

        return "\n\n".join(parts)