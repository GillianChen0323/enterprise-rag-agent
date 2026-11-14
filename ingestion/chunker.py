"""Adaptive chunking with fixed-size, sentence-aware, and hierarchical strategies.
Hierarchical chunking uses smaller child chunks for retrieval and larger parent chunks
for generation, balancing retrieval granularity with context preservation."""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkedDocument:
    """A document chunk."""
    chunk_id: str           # Content-hash identifier.
    page_content: str       # Child content for retrieval.
    parent_text: str        # Parent content for generation.
    parent_id: str          # Parent identifier.
    metadata: dict

    @classmethod
    def from_texts(cls, child_text: str, parent_text: str,
                   parent_id: str, metadata: dict) -> "ChunkedDocument":
        chunk_id = hashlib.md5(child_text.encode()).hexdigest()[:12]
        return cls(
            chunk_id=chunk_id,
            page_content=child_text,
            parent_text=parent_text,
            parent_id=parent_id,
            metadata=metadata,
        )


class AdaptiveChunker:
    """Chunker supporting three configurable strategies."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ─────────────────────────────────────────────
    # Fixed-size baseline.
    # ─────────────────────────────────────────────
    def fixed_chunk(
        self, text: str, metadata: dict = None
    ) -> list[ChunkedDocument]:
        """Split by a fixed character count without semantic boundary detection. Retained
        as a comparison baseline."""
        chunks = []
        start = 0
        parent_id = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(ChunkedDocument.from_texts(
                child_text=chunk_text,
                parent_text=chunk_text,  # Parent and child content are identical.
                parent_id=parent_id,
                metadata=metadata or {},
            ))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ─────────────────────────────────────────────
    # Sentence-aware strategy.
    # ─────────────────────────────────────────────
    def sentence_chunk(
        self, text: str, metadata: dict = None
    ) -> list[ChunkedDocument]:
        """Split at sentence boundaries to reduce sentence truncation."""
        # Chinese and English sentence boundaries.
        import re
        sentences = re.split(r'(?<=[。！？\.\!\?])\s*|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current = []
        current_len = 0
        parent_id = hashlib.md5(text[:100].encode()).hexdigest()[:8]

        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(ChunkedDocument.from_texts(
                    child_text=chunk_text,
                    parent_text=chunk_text,
                    parent_id=parent_id,
                    metadata=metadata or {},
                ))
                # Retain the last sentence as overlap.
                current = current[-1:]
                current_len = len(current[0]) if current else 0
            current.append(sent)
            current_len += len(sent)

        if current:
            chunk_text = " ".join(current)
            chunks.append(ChunkedDocument.from_texts(
                child_text=chunk_text,
                parent_text=chunk_text,
                parent_id=parent_id,
                metadata=metadata or {},
            ))
        return chunks

    # ─────────────────────────────────────────────
    # Hierarchical strategy.
    # ─────────────────────────────────────────────
    def hierarchical_chunk(
        self,
        text: str,
        metadata: dict = None,
        parent_chunk_size: int = 1024,
        parent_overlap: int = 128,
        child_chunk_size: int = 256,
        child_overlap: int = 32,
    ) -> list[ChunkedDocument]:
        """Create parent and child chunks. Retrieve using smaller child chunks and supply
        associated parent content to generation to preserve surrounding context."""
        # Split parent chunks.
        parent_chunks = self._split_text(
            text, chunk_size=parent_chunk_size, overlap=parent_overlap
        )

        all_child_chunks = []
        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = hashlib.md5(
                f"{p_idx}:{parent_text[:50]}".encode()
            ).hexdigest()[:10]

            # Split children within each parent.
            child_texts = self._split_text(
                parent_text, chunk_size=child_chunk_size, overlap=child_overlap
            )

            for c_idx, child_text in enumerate(child_texts):
                child_meta = {
                    **(metadata or {}),
                    "parent_id": parent_id,
                    "parent_idx": p_idx,
                    "child_idx": c_idx,
                    "total_parents": len(parent_chunks),
                }
                chunk = ChunkedDocument.from_texts(
                    child_text=child_text,
                    parent_text=parent_text,   # Store parent context for generation.
                    parent_id=parent_id,
                    metadata=child_meta,
                )
                all_child_chunks.append(chunk)

        logger.info(
            f"Hierarchical chunking: {len(parent_chunks)} parents → "
            f"{len(all_child_chunks)} children | doc_len={len(text)}"
        )
        return all_child_chunks

    # ─────────────────────────────────────────────
    # Recursive splitting.
    # ─────────────────────────────────────────────
    @staticmethod
    def _split_text(
        text: str, chunk_size: int, overlap: int
    ) -> list[str]:
        """Recursively split by prioritized separators: paragraphs, newlines, Chinese
        sentence boundaries, English sentence boundaries, and spaces."""
        separators = ["\n\n", "\n", "。", ".", " ", ""]
        chunks = []

        def split_recursive(t: str, seps: list[str]) -> list[str]:
            if not seps or len(t) <= chunk_size:
                return [t] if t.strip() else []
            sep = seps[0]
            parts = t.split(sep) if sep else list(t)
            result = []
            current = ""
            for part in parts:
                trial = current + (sep if current else "") + part
                if len(trial) <= chunk_size:
                    current = trial
                else:
                    if current:
                        result.append(current)
                    if len(part) > chunk_size:
                        result.extend(split_recursive(part, seps[1:]))
                        current = ""
                    else:
                        current = part
            if current:
                result.append(current)
            return result

        raw_chunks = split_recursive(text, separators)

        # Overlap adjacent chunks.
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and overlap > 0:
                prev = raw_chunks[i - 1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                chunk = overlap_text + chunk
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks

    # ─────────────────────────────────────────────
    # Strategy dispatcher.
    # ─────────────────────────────────────────────
    def chunk(
        self,
        text: str,
        metadata: dict = None,
        strategy: str = "hierarchical",
    ) -> list[ChunkedDocument]:
        """Dispatch the fixed, sentence, or hierarchical strategy."""
        if strategy == "hierarchical":
            return self.hierarchical_chunk(
                text, metadata,
                parent_chunk_size=self.chunk_size * 2,
                parent_overlap=self.chunk_overlap * 2,
                child_chunk_size=self.chunk_size // 2,
                child_overlap=self.chunk_overlap // 2,
            )
        elif strategy == "sentence":
            return self.sentence_chunk(text, metadata)
        else:
            return self.fixed_chunk(text, metadata)
