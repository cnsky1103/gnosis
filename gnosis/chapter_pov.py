import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from .models import ChapterPovEntry


@dataclass(frozen=True)
class ChapterTitleMatch:
    entry: ChapterPovEntry
    start: int


@dataclass(frozen=True)
class ChapterSegment:
    text: str
    chapter_title: Optional[str] = None
    pov_speaker: Optional[str] = None


def normalize_chapter_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "").strip())


def normalize_text_line_endings(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _entry_from_raw(
    raw: object, warnings: List[str], index: int
) -> Optional[ChapterPovEntry]:
    if not isinstance(raw, dict):
        warnings.append(f"chapter_pov entry is not an object at index {index}")
        return None

    try:
        entry = ChapterPovEntry.model_validate(raw)
    except Exception as exc:
        warnings.append(f"invalid chapter_pov entry at index {index}: {exc}")
        return None

    if not entry.chapter_title.strip():
        warnings.append(f"chapter_pov entry missing chapter_title at index {index}")
        return None
    if not entry.pov_speaker.strip():
        warnings.append(f"chapter_pov entry missing pov_speaker at index {index}")
        return None
    return entry


def load_chapter_pov_entries(path: str) -> Tuple[List[ChapterPovEntry], List[str]]:
    if not path or not os.path.exists(path):
        return [], [f"chapter_pov.json missing: {path}"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        return [], [f"chapter_pov.json invalid JSON: {path}: {exc}"]
    except OSError as exc:
        return [], [f"chapter_pov.json could not be read: {path}: {exc}"]

    if not isinstance(payload, list):
        return [], [f"chapter_pov.json root must be an array: {path}"]

    entries: List[ChapterPovEntry] = []
    warnings: List[str] = []
    for index, raw in enumerate(payload):
        entry = _entry_from_raw(raw, warnings, index)
        if entry is not None:
            entries.append(entry)

    return entries, warnings


class ChapterPovStore:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> Tuple[List[ChapterPovEntry], List[str]]:
        return load_chapter_pov_entries(self.path)

    def save(self, entries: Sequence[ChapterPovEntry]) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                [entry.model_dump() for entry in entries],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def merge_extracted(
        self, extracted_entries: Iterable[ChapterPovEntry]
    ) -> List[ChapterPovEntry]:
        existing_entries, _warnings = self.load()
        merged: List[ChapterPovEntry] = list(existing_entries)
        seen_titles = {
            normalize_chapter_title(entry.chapter_title) for entry in existing_entries
        }

        for entry in extracted_entries:
            normalized_title = normalize_chapter_title(entry.chapter_title)
            if not normalized_title or normalized_title in seen_titles:
                continue
            merged.append(entry)
            seen_titles.add(normalized_title)

        self.save(merged)
        return merged


def find_chapter_title_matches(
    text: str, entries: Sequence[ChapterPovEntry]
) -> Tuple[List[ChapterTitleMatch], List[str]]:
    normalized_text = normalize_text_line_endings(text)
    matches: List[ChapterTitleMatch] = []
    warnings: List[str] = []
    search_start = 0

    for entry in entries:
        title = normalize_text_line_endings(entry.chapter_title)
        if not title.strip():
            warnings.append("chapter entry missing chapter_title")
            continue

        first_start = normalized_text.find(title)
        start = normalized_text.find(title, search_start)

        if start == -1:
            if first_start != -1 and first_start < search_start:
                warnings.append(f"chapter title out of order: {entry.chapter_title}")
            else:
                warnings.append(f"chapter title not found: {entry.chapter_title}")
            continue

        duplicate_start = normalized_text.find(title, start + len(title))
        if duplicate_start != -1:
            warnings.append(f"chapter title matched more than once: {entry.chapter_title}")

        matches.append(ChapterTitleMatch(entry=entry, start=start))
        search_start = start + len(title)

    return matches, warnings


def build_chapter_segments(
    text: str, matches: Sequence[ChapterTitleMatch]
) -> List[ChapterSegment]:
    normalized_text = normalize_text_line_endings(text)
    if not matches:
        return [ChapterSegment(text=normalized_text)]

    segments: List[ChapterSegment] = []
    first_start = matches[0].start
    if first_start > 0:
        segments.append(ChapterSegment(text=normalized_text[:first_start]))

    for index, match in enumerate(matches):
        end = matches[index + 1].start if index + 1 < len(matches) else len(normalized_text)
        segment_text = normalized_text[match.start : end]
        if segment_text:
            segments.append(
                ChapterSegment(
                    text=segment_text,
                    chapter_title=match.entry.chapter_title,
                    pov_speaker=match.entry.pov_speaker,
                )
            )

    return segments


def validate_chapter_pov_entries(
    entries: Sequence[ChapterPovEntry], known_character_names: Set[str]
) -> List[str]:
    warnings: List[str] = []

    for entry in entries:
        if not entry.chapter_title.strip():
            warnings.append("chapter_pov entry missing chapter_title")
            continue
        if not entry.pov_speaker.strip():
            warnings.append(f"chapter_pov entry missing pov_speaker: {entry.chapter_title}")
            continue
        if entry.pov_speaker not in known_character_names:
            warnings.append(
                "chapter POV speaker not in character_db.json: "
                f"{entry.pov_speaker} ({entry.chapter_title})"
            )

    return warnings
