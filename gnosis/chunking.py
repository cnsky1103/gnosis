from dataclasses import dataclass
from typing import List
import re


@dataclass(frozen=True)
class ChunkingConfig:
    target_chars: int = 1800
    min_chars: int = 1200
    max_chars: int = 2600
    rolling_paragraphs: int = 2
    rolling_max_chars: int = 500


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    paragraphs: List[str]


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_paragraphs(text: str) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p.strip()]


def _split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentence_parts = [p for p in re.split(r"(?<=[。！？!?；;])", paragraph) if p]
    if not sentence_parts:
        sentence_parts = [paragraph]

    segments: List[str] = []
    current = ""
    for sentence in sentence_parts:
        sentence = sentence.strip()
        if not sentence:
            continue

        projected_len = len(current) + len(sentence)
        if current and projected_len > max_chars:
            segments.append(current)
            current = sentence
        else:
            current += sentence

    if current:
        segments.append(current)

    final_segments: List[str] = []
    for segment in segments:
        if len(segment) <= max_chars:
            final_segments.append(segment)
            continue
        for i in range(0, len(segment), max_chars):
            final_segments.append(segment[i : i + max_chars])

    return [s for s in final_segments if s]


def split_text_into_chunks(text: str, config: ChunkingConfig) -> List[TextChunk]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    expanded_paragraphs: List[str] = []
    for paragraph in paragraphs:
        expanded_paragraphs.extend(_split_long_paragraph(paragraph, config.max_chars))

    chunk_paragraphs: List[List[str]] = []
    current: List[str] = []
    current_len = 0

    for paragraph in expanded_paragraphs:
        paragraph_len = len(paragraph)
        separator_len = 2 if current else 0
        projected_len = current_len + separator_len + paragraph_len

        should_flush = bool(current) and (
            projected_len > config.max_chars
            or (current_len >= config.target_chars and projected_len > config.target_chars)
        )
        if should_flush:
            chunk_paragraphs.append(current)
            current = [paragraph]
            current_len = paragraph_len
            continue

        if current:
            current_len += 2
        current.append(paragraph)
        current_len += paragraph_len

    if current:
        chunk_paragraphs.append(current)

    if len(chunk_paragraphs) >= 2:
        last_text_len = len("\n\n".join(chunk_paragraphs[-1]))
        if last_text_len < config.min_chars:
            chunk_paragraphs[-2].extend(chunk_paragraphs[-1])
            chunk_paragraphs.pop()

    chunks: List[TextChunk] = []
    for idx, chunk in enumerate(chunk_paragraphs, start=1):
        chunks.append(TextChunk(index=idx, paragraphs=chunk, text="\n\n".join(chunk)))
    return chunks


def build_rolling_context(chunk: TextChunk, config: ChunkingConfig) -> str:
    if not chunk.paragraphs:
        return ""
    tail = chunk.paragraphs[-config.rolling_paragraphs :]
    tail_text = "\n\n".join(tail)
    if len(tail_text) <= config.rolling_max_chars:
        return tail_text
    return tail_text[-config.rolling_max_chars :]
