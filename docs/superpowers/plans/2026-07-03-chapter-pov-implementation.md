# Chapter POV Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reviewed chapter POV metadata so pass2 splits novels by chapter first and preserves first-person narrator context across chunks.

**Architecture:** Pass1 extracts characters and chapter POV entries in the same LLM response, then persists chapter POV entries to a new reviewed artifact, `chapter_pov.json`. Pass2 loads that reviewed artifact, uses exact chapter-title string matching to build chapter segments, chunks inside each chapter, and injects chapter title plus POV speaker into every pass2 prompt. All missing or imperfect chapter metadata produces warnings only.

**Tech Stack:** Python 3.8+, Pydantic models, existing OpenAI-compatible LLM client, pytest, existing `gnosis` pipeline modules.

---

## File Structure

Create:

- `gnosis/chapter_pov.py`
  - Owns `chapter_pov.json` loading, saving, deduplication, title matching, warning generation, and `ChapterSegment` objects.
- `tests/test_chapter_pov.py`
  - Tests persistence, deduplication, title matching, warning-only behavior, and speaker validation.
- `tests/test_chunking.py`
  - Tests chapter metadata propagation through chunking.
- `tests/test_pipeline_chapter_pov.py`
  - Tests pass1 persistence wiring and pass2 prompt metadata wiring without making real LLM calls.

Modify:

- `gnosis/models.py`
  - Add `ChapterPovEntry`, `ChapterPovExtraction`, and `CharacterExtraction.chapters`.
- `gnosis/chunking.py`
  - Add optional chapter metadata to `TextChunk`.
  - Add `split_chapter_segments_into_chunks()`.
- `gnosis/llm_director.py`
  - Update pass1 output schema and extraction instructions.
  - Add pass2 current chapter metadata section.
- `gnosis/pipeline.py`
  - Collect `extraction.chapters` in pass1 and persist them.
  - Load chapter POV metadata in pass2 and build chapter-first chunks.
  - Pass `chapter_title` and `pov_speaker` into pass2 prompt formatting.
- `main.py`
  - Define `chapter_pov_path`.
  - Pass that path to `run_pass1()` and `run_pass2()`.

Do not modify:

- `character_db.json` top-level format.
- TTS, merge, QA, or proofread behavior.

---

### Task 1: Add Chapter POV Models

**Files:**
- Modify: `gnosis/models.py`
- Test: `tests/test_chapter_pov.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_chapter_pov.py` with these initial tests:

```python
from gnosis.models import CharacterExtraction, ChapterPovEntry


def test_character_extraction_accepts_old_response_without_chapters():
    payload = {
        "new_characters": [
            {
                "name": "浅村悠太",
                "gender": "male",
                "voice_archetype": "男-普通",
                "description": "第一人称主角",
            }
        ]
    }

    extraction = CharacterExtraction.model_validate(payload)

    assert extraction.new_characters[0].name == "浅村悠太"
    assert extraction.chapters == []


def test_character_extraction_accepts_chapter_pov_entries():
    payload = {
        "new_characters": [],
        "chapters": [
            {
                "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
                "pov_speaker": "绫濑沙季",
                "evidence": "标题末尾出现绫濑沙季",
            }
        ],
    }

    extraction = CharacterExtraction.model_validate(payload)

    assert extraction.chapters == [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="绫濑沙季",
            evidence="标题末尾出现绫濑沙季",
        )
    ]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_chapter_pov.py -v
```

Expected: fail with an import error for `ChapterPovEntry` or an attribute error for `CharacterExtraction.chapters`.

- [ ] **Step 3: Implement models**

Replace `gnosis/models.py` with:

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class CharacterProfile(BaseModel):
    name: str
    gender: Literal["male", "female", "unknown"]
    voice_archetype: str = "未知"
    # 绑定后的声线种子ID，例如 "zuole"
    voice: Optional[str] = None
    ref_audio_path: Optional[str] = None
    ref_audio_text: Optional[str] = None
    description: Optional[str] = None


class ChapterPovEntry(BaseModel):
    chapter_title: str
    pov_speaker: str
    evidence: Optional[str] = None


class ChapterPovExtraction(BaseModel):
    chapters: List[ChapterPovEntry] = Field(default_factory=list)


class ScriptLine(BaseModel):
    text: str
    speaker: str
    #emotion: str = "neutral"


class CharacterExtraction(BaseModel):
    new_characters: List[CharacterProfile]
    chapters: List[ChapterPovEntry] = Field(default_factory=list)


class ScriptResult(BaseModel):
    script: List[ScriptLine]
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_chapter_pov.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add gnosis/models.py tests/test_chapter_pov.py
git commit -m "feat: add chapter POV extraction models"
```

---

### Task 2: Add Chapter POV Persistence and Segmentation

**Files:**
- Create: `gnosis/chapter_pov.py`
- Modify: `tests/test_chapter_pov.py`

- [ ] **Step 1: Add failing persistence and segmentation tests**

Append these tests to `tests/test_chapter_pov.py`:

```python
import json

from gnosis.chapter_pov import (
    ChapterPovStore,
    build_chapter_segments,
    find_chapter_title_matches,
    load_chapter_pov_entries,
    validate_chapter_pov_entries,
)
from gnosis.models import ChapterPovEntry


def test_chapter_pov_store_preserves_reviewed_entries(tmp_path):
    path = tmp_path / "chapter_pov.json"
    path.write_text(
        json.dumps(
            [
                {
                    "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
                    "pov_speaker": "人工修正后的绫濑沙季",
                    "evidence": "人工审核",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = ChapterPovStore(str(path))

    entries = store.merge_extracted(
        [
            ChapterPovEntry(
                chapter_title="第四卷 9月3日（星期四）绫濑沙季",
                pov_speaker="绫濑沙季",
                evidence="LLM 抽取",
            )
        ]
    )

    assert entries == [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="人工修正后的绫濑沙季",
            evidence="人工审核",
        )
    ]
    assert load_chapter_pov_entries(str(path))[0] == entries


def test_chapter_pov_store_deduplicates_collapsed_whitespace(tmp_path):
    path = tmp_path / "chapter_pov.json"
    store = ChapterPovStore(str(path))

    entries = store.merge_extracted(
        [
            ChapterPovEntry(
                chapter_title="第四卷   9月3日（星期四）  绫濑沙季",
                pov_speaker="绫濑沙季",
            ),
            ChapterPovEntry(
                chapter_title="第四卷 9月3日（星期四） 绫濑沙季",
                pov_speaker="错误的重复项",
            ),
        ]
    )

    assert len(entries) == 1
    assert entries[0].pov_speaker == "绫濑沙季"


def test_find_chapter_title_matches_returns_ordered_offsets():
    text = (
        "序章内容\n\n"
        "第四卷 9月3日（星期四）浅村悠太\n正文一\n\n"
        "第四卷 9月3日（星期四）绫濑沙季\n正文二"
    )
    entries = [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        ),
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="绫濑沙季",
        ),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)

    assert warnings == []
    assert [match.entry.pov_speaker for match in matches] == ["浅村悠太", "绫濑沙季"]
    assert text[matches[0].start :].startswith("第四卷 9月3日（星期四）浅村悠太")
    assert text[matches[1].start :].startswith("第四卷 9月3日（星期四）绫濑沙季")


def test_missing_chapter_title_warns_and_does_not_stop_segmentation():
    text = "第四卷 9月3日（星期四）浅村悠太\n正文一"
    entries = [
        ChapterPovEntry(
            chapter_title="不存在的章节",
            pov_speaker="绫濑沙季",
        ),
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        ),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)

    assert len(matches) == 1
    assert "chapter title not found: 不存在的章节" in warnings


def test_out_of_order_chapter_title_warns_and_is_skipped():
    text = (
        "第四卷 9月3日（星期四）浅村悠太\n正文一\n\n"
        "第四卷 9月3日（星期四）绫濑沙季\n正文二"
    )
    entries = [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="绫濑沙季",
        ),
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        ),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)

    assert [match.entry.pov_speaker for match in matches] == ["绫濑沙季"]
    assert "chapter title out of order: 第四卷 9月3日（星期四）浅村悠太" in warnings


def test_text_before_first_match_becomes_no_pov_segment():
    text = "序章内容\n\n第四卷 9月3日（星期四）浅村悠太\n正文一"
    entries = [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        )
    ]

    matches, warnings = find_chapter_title_matches(text, entries)
    segments = build_chapter_segments(text, matches)

    assert warnings == []
    assert segments[0].chapter_title is None
    assert segments[0].pov_speaker is None
    assert segments[0].text == "序章内容\n\n"
    assert segments[1].chapter_title == "第四卷 9月3日（星期四）浅村悠太"
    assert segments[1].pov_speaker == "浅村悠太"


def test_validate_chapter_pov_entries_warns_for_unknown_speaker():
    entries = [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="不存在的人",
        )
    ]

    warnings = validate_chapter_pov_entries(entries, {"浅村悠太"})

    assert warnings == [
        "chapter POV speaker not in character_db.json: 不存在的人 (第四卷 9月3日（星期四）浅村悠太)"
    ]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_chapter_pov.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'gnosis.chapter_pov'`.

- [ ] **Step 3: Implement `gnosis/chapter_pov.py`**

Create `gnosis/chapter_pov.py`:

```python
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


def _entry_from_raw(raw: object, warnings: List[str], index: int) -> Optional[ChapterPovEntry]:
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

    def merge_extracted(self, extracted_entries: Iterable[ChapterPovEntry]) -> List[ChapterPovEntry]:
        existing_entries, warnings = self.load()
        for warning in warnings:
            if "missing" not in warning:
                print(f"⚠️ {warning}")

        merged: List[ChapterPovEntry] = list(existing_entries)
        seen = {normalize_chapter_title(entry.chapter_title) for entry in merged}

        for entry in extracted_entries:
            key = normalize_chapter_title(entry.chapter_title)
            if not key or key in seen:
                continue
            merged.append(entry)
            seen.add(key)

        self.save(merged)
        return merged


def find_chapter_title_matches(
    text: str, entries: Sequence[ChapterPovEntry]
) -> Tuple[List[ChapterTitleMatch], List[str]]:
    normalized_text = normalize_text_line_endings(text)
    warnings: List[str] = []
    matches: List[ChapterTitleMatch] = []
    search_start = 0

    for entry in entries:
        title = normalize_text_line_endings(entry.chapter_title)
        if not title.strip():
            warnings.append("chapter entry missing chapter_title")
            continue

        first_anywhere = normalized_text.find(title)
        match_start = normalized_text.find(title, search_start)

        if match_start == -1:
            if first_anywhere != -1 and first_anywhere < search_start:
                warnings.append(f"chapter title out of order: {entry.chapter_title}")
            else:
                warnings.append(f"chapter title not found: {entry.chapter_title}")
            continue

        next_duplicate = normalized_text.find(title, match_start + len(title))
        if next_duplicate != -1:
            warnings.append(f"chapter title matched more than once: {entry.chapter_title}")

        matches.append(ChapterTitleMatch(entry=entry, start=match_start))
        search_start = match_start + len(title)

    return matches, warnings


def build_chapter_segments(text: str, matches: Sequence[ChapterTitleMatch]) -> List[ChapterSegment]:
    normalized_text = normalize_text_line_endings(text)
    if not matches:
        return [ChapterSegment(text=normalized_text)]

    segments: List[ChapterSegment] = []
    first_start = matches[0].start
    if first_start > 0:
        segments.append(ChapterSegment(text=normalized_text[:first_start]))

    for index, match in enumerate(matches):
        end = matches[index + 1].start if index + 1 < len(matches) else len(normalized_text)
        segment_text = normalized_text[match.start:end]
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
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_chapter_pov.py -v
```

Expected: all tests in `tests/test_chapter_pov.py` pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add gnosis/chapter_pov.py tests/test_chapter_pov.py
git commit -m "feat: add chapter POV persistence and matching"
```

---

### Task 3: Add Chapter-Aware Chunking

**Files:**
- Modify: `gnosis/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write failing chunking tests**

Create `tests/test_chunking.py`:

```python
from gnosis.chapter_pov import ChapterSegment
from gnosis.chunking import ChunkingConfig, split_chapter_segments_into_chunks


def test_split_chapter_segments_keeps_chunks_inside_chapter_boundaries():
    segments = [
        ChapterSegment(
            chapter_title="第一章 浅村悠太",
            pov_speaker="浅村悠太",
            text="第一章 浅村悠太\n\n" + "\n\n".join([f"悠太段落{i}" for i in range(8)]),
        ),
        ChapterSegment(
            chapter_title="第二章 绫濑沙季",
            pov_speaker="绫濑沙季",
            text="第二章 绫濑沙季\n\n" + "\n\n".join([f"沙季段落{i}" for i in range(8)]),
        ),
    ]
    config = ChunkingConfig(target_chars=20, min_chars=1, max_chars=35)

    chunks = split_chapter_segments_into_chunks(segments, config)

    assert len(chunks) > 2
    for chunk in chunks:
        assert not ("第一章 浅村悠太" in chunk.text and "第二章 绫濑沙季" in chunk.text)
    assert {chunk.pov_speaker for chunk in chunks if "悠太段落" in chunk.text} == {"浅村悠太"}
    assert {chunk.pov_speaker for chunk in chunks if "沙季段落" in chunk.text} == {"绫濑沙季"}


def test_split_chapter_segments_preserves_no_pov_segment():
    segments = [
        ChapterSegment(text="序章内容\n\n没有章节元数据"),
        ChapterSegment(
            chapter_title="第一章 浅村悠太",
            pov_speaker="浅村悠太",
            text="第一章 浅村悠太\n\n正文",
        ),
    ]
    config = ChunkingConfig(target_chars=50, min_chars=1, max_chars=80)

    chunks = split_chapter_segments_into_chunks(segments, config)

    assert chunks[0].chapter_title is None
    assert chunks[0].pov_speaker is None
    assert chunks[1].chapter_title == "第一章 浅村悠太"
    assert chunks[1].pov_speaker == "浅村悠太"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_chunking.py -v
```

Expected: fail with import error for `split_chapter_segments_into_chunks` or missing `TextChunk.pov_speaker`.

- [ ] **Step 3: Implement chapter metadata in chunking**

Modify `gnosis/chunking.py`:

```python
from dataclasses import dataclass
from typing import List, Optional
import re

from .chapter_pov import ChapterSegment
```

Update `TextChunk`:

```python
@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    paragraphs: List[str]
    chapter_title: Optional[str] = None
    pov_speaker: Optional[str] = None
```

Add this helper after `split_text_into_chunks()`:

```python
def split_chapter_segments_into_chunks(
    segments: List[ChapterSegment], config: ChunkingConfig
) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    next_index = 1

    for segment in segments:
        segment_chunks = split_text_into_chunks(segment.text, config)
        for chunk in segment_chunks:
            chunks.append(
                TextChunk(
                    index=next_index,
                    text=chunk.text,
                    paragraphs=chunk.paragraphs,
                    chapter_title=segment.chapter_title,
                    pov_speaker=segment.pov_speaker,
                )
            )
            next_index += 1

    return chunks
```

Keep existing `split_text_into_chunks()` behavior unchanged for callers that do not use chapter segmentation.

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_chunking.py tests/test_chapter_pov.py -v
```

Expected: both files pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add gnosis/chunking.py tests/test_chunking.py
git commit -m "feat: add chapter-aware chunking"
```

---

### Task 4: Update LLM Prompts for Chapter POV

**Files:**
- Modify: `gnosis/llm_director.py`
- Test: `tests/test_pipeline_chapter_pov.py`

- [ ] **Step 1: Write failing prompt tests**

Create `tests/test_pipeline_chapter_pov.py`:

```python
from gnosis.llm_director import PASS1_PROMPT_TEMPLATE, PASS2_PROMPT_TEMPLATE


def test_pass1_prompt_requests_chapter_pov_output():
    prompt = PASS1_PROMPT_TEMPLATE.format(
        known_characters_str="无",
        allowed_character_tags="男-普通\n女-普通",
        project_pass1_prompt="无",
    )

    assert '"chapters"' in prompt
    assert "chapter_title" in prompt
    assert "pov_speaker" in prompt
    assert "章节" in prompt


def test_pass2_prompt_includes_current_chapter_metadata():
    prompt = PASS2_PROMPT_TEMPLATE.format(
        available_characters_str="- 绫濑沙季 (female)",
        previous_chunk_context_str="上一段",
        chunk_index=1,
        total_chunks=1,
        chapter_title="第四卷 9月3日（星期四）绫濑沙季",
        pov_speaker="绫濑沙季",
        project_pass2_prompt="无",
    )

    assert "当前章节信息" in prompt
    assert "第四卷 9月3日（星期四）绫濑沙季" in prompt
    assert "第一人称视角：绫濑沙季" in prompt
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_pipeline_chapter_pov.py -v
```

Expected: first test fails because pass1 prompt lacks chapter POV schema; second test fails because pass2 template lacks `chapter_title` and `pov_speaker` placeholders.

- [ ] **Step 3: Update pass1 prompt**

In `gnosis/llm_director.py`, update the pass1 task and output sections so the template includes:

```text
# 章节与视角提取任务
6. 如果片段中出现章节标题、日期标题、日记标题，或其他强章节边界，请同时提取章节 POV 信息到 `chapters`。
7. `chapter_title` 必须逐字复制原文中的章节标题；`pov_speaker` 是该章节第一人称旁白所属角色；`evidence` 简短说明判断依据。
8. 如果无法判断章节 POV，不要猜测，省略该章节条目。
```

Replace the pass1 output example with:

```text
请以**紧凑的** JSON 格式输出（单行、无换行、无缩进、键值之间不加多余空格），不要包含 markdown 标记。示例：
{{"new_characters":[{{"name":"...","gender":"...","voice_archetype":"...","description":"..."}}],"chapters":[{{"chapter_title":"...","pov_speaker":"...","evidence":"..."}}]}}
```

- [ ] **Step 4: Update pass2 prompt**

In `PASS2_PROMPT_TEMPLATE`, insert this section after current progress and before previous context:

```text
# 当前章节信息
章节标题：{chapter_title}
第一人称视角：{pov_speaker}

如果“第一人称视角”不是“未提供”，当前小说片段中所有非对话旁白、第一人称心理活动、第一人称叙述句的 `speaker` 必须使用该角色名。
```

Do not remove the existing previous-context section.

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
pytest tests/test_pipeline_chapter_pov.py -v
```

Expected: both prompt tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add gnosis/llm_director.py tests/test_pipeline_chapter_pov.py
git commit -m "feat: add chapter POV prompt context"
```

---

### Task 5: Wire Pass1 Chapter POV Persistence

**Files:**
- Modify: `gnosis/pipeline.py`
- Modify: `tests/test_pipeline_chapter_pov.py`

- [ ] **Step 1: Add failing pass1 persistence test**

Append this test to `tests/test_pipeline_chapter_pov.py`:

```python
import json

from gnosis.chunking import ChunkingConfig
from gnosis.pipeline import run_pass1
from gnosis.state_manager import CharacterManager


def test_run_pass1_persists_extracted_chapter_pov(tmp_path, monkeypatch):
    character_db_path = tmp_path / "character_db.json"
    chapter_pov_path = tmp_path / "chapter_pov.json"
    manager = CharacterManager(db_path=str(character_db_path))

    def fake_get_raw_response(**kwargs):
        return (
            json.dumps(
                {
                    "new_characters": [
                        {
                            "name": "绫濑沙季",
                            "gender": "female",
                            "voice_archetype": "女-普通",
                            "description": "第一人称视角角色",
                        }
                    ],
                    "chapters": [
                        {
                            "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
                            "pov_speaker": "绫濑沙季",
                            "evidence": "标题末尾出现绫濑沙季",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            str(tmp_path / "cache.json"),
            False,
        )

    monkeypatch.setattr("gnosis.pipeline._get_raw_response", fake_get_raw_response)

    run_pass1(
        "第四卷 9月3日（星期四）绫濑沙季\n\n我走进教室。",
        manager,
        ChunkingConfig(target_chars=1000, min_chars=1, max_chars=1200),
        cache_dir=str(tmp_path / "cache"),
        chapter_pov_path=str(chapter_pov_path),
    )

    payload = json.loads(chapter_pov_path.read_text(encoding="utf-8"))
    assert payload == [
        {
            "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
            "pov_speaker": "绫濑沙季",
            "evidence": "标题末尾出现绫濑沙季",
        }
    ]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_pipeline_chapter_pov.py::test_run_pass1_persists_extracted_chapter_pov -v
```

Expected: fail because `run_pass1()` does not accept `chapter_pov_path`.

- [ ] **Step 3: Update `run_pass1()` signature and persistence**

In `gnosis/pipeline.py`, update imports:

```python
from .chapter_pov import ChapterPovStore
from .models import CharacterExtraction, ChapterPovEntry, ScriptResult
```

Update `run_pass1()` signature:

```python
def run_pass1(
    text_segment,
    char_manager,
    chunking_config: ChunkingConfig = None,
    cache_dir: str = "data/llm_cache",
    pass1_custom_prompt: str = "",
    chapter_pov_path: Optional[str] = None,
):
```

Inside `run_pass1()`, before the chunk loop:

```python
    extracted_chapters: List[ChapterPovEntry] = []
```

After adding characters from each extraction:

```python
        extracted_chapters.extend(extraction.chapters)
```

After `char_manager.load_db()`:

```python
    if chapter_pov_path:
        chapter_store = ChapterPovStore(chapter_pov_path)
        merged_chapters = chapter_store.merge_extracted(extracted_chapters)
        print(f"✅ 章节 POV 已更新: {len(merged_chapters)} 条")
        if not extracted_chapters:
            print("⚠️ pass1 未提取到章节 POV；如果本书有多 POV 章节，请检查 pass1 缓存或 prompt")
```

- [ ] **Step 4: Run test and verify pass**

Run:

```bash
pytest tests/test_pipeline_chapter_pov.py::test_run_pass1_persists_extracted_chapter_pov -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add gnosis/pipeline.py tests/test_pipeline_chapter_pov.py
git commit -m "feat: persist chapter POV from pass1"
```

---

### Task 6: Wire Pass2 Chapter-First Chunking and Prompt Metadata

**Files:**
- Modify: `gnosis/pipeline.py`
- Modify: `tests/test_pipeline_chapter_pov.py`

- [ ] **Step 1: Add failing pass2 wiring tests**

Append these tests to `tests/test_pipeline_chapter_pov.py`:

```python
from gnosis.models import CharacterProfile


def test_run_pass2_injects_chapter_pov_into_prompt(tmp_path, monkeypatch):
    chapter_pov_path = tmp_path / "chapter_pov.json"
    chapter_pov_path.write_text(
        json.dumps(
            [
                {
                    "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
                    "pov_speaker": "绫濑沙季",
                    "evidence": "人工审核",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    character_db_path = tmp_path / "character_db.json"
    manager = CharacterManager(db_path=str(character_db_path))
    manager.add_character(
        CharacterProfile(
            name="绫濑沙季",
            gender="female",
            voice_archetype="女-普通",
            description="第一人称视角角色",
        )
    )

    captured_system_prompts = []

    def fake_get_raw_response(**kwargs):
        captured_system_prompts.append(kwargs["messages"][0]["content"])
        return (json.dumps({"script": []}, ensure_ascii=False), str(tmp_path / "cache.json"), False)

    monkeypatch.setattr("gnosis.pipeline._get_raw_response", fake_get_raw_response)

    run_pass2(
        "序章\n\n第四卷 9月3日（星期四）绫濑沙季\n\n我走进教室。\n\n我坐下。",
        manager,
        ChunkingConfig(target_chars=10, min_chars=1, max_chars=30),
        cache_dir=str(tmp_path / "cache"),
        pass2_workers=1,
        chapter_pov_path=str(chapter_pov_path),
    )

    assert any("第一人称视角：绫濑沙季" in prompt for prompt in captured_system_prompts)
    assert any("章节标题：第四卷 9月3日（星期四）绫濑沙季" in prompt for prompt in captured_system_prompts)


def test_run_pass2_warns_but_continues_without_chapter_pov_file(tmp_path, monkeypatch, capsys):
    character_db_path = tmp_path / "character_db.json"
    manager = CharacterManager(db_path=str(character_db_path))
    manager.add_character(
        CharacterProfile(
            name="浅村悠太",
            gender="male",
            voice_archetype="男-普通",
            description="第一人称视角角色",
        )
    )

    def fake_get_raw_response(**kwargs):
        assert "第一人称视角：未提供" in kwargs["messages"][0]["content"]
        return (json.dumps({"script": []}, ensure_ascii=False), str(tmp_path / "cache.json"), False)

    monkeypatch.setattr("gnosis.pipeline._get_raw_response", fake_get_raw_response)

    result = run_pass2(
        "没有章节文件也要继续。",
        manager,
        ChunkingConfig(target_chars=1000, min_chars=1, max_chars=1200),
        cache_dir=str(tmp_path / "cache"),
        pass2_workers=1,
        chapter_pov_path=str(tmp_path / "missing_chapter_pov.json"),
    )

    captured = capsys.readouterr()
    assert "chapter_pov.json missing" in captured.out
    assert result["script"] == []
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_pipeline_chapter_pov.py::test_run_pass2_injects_chapter_pov_into_prompt tests/test_pipeline_chapter_pov.py::test_run_pass2_warns_but_continues_without_chapter_pov_file -v
```

Expected: fail because `run_pass2()` does not accept `chapter_pov_path`, and pass2 prompt formatting does not pass chapter metadata.

- [ ] **Step 3: Update imports**

In `gnosis/pipeline.py`, update chunking and chapter imports:

```python
from .chapter_pov import (
    build_chapter_segments,
    find_chapter_title_matches,
    load_chapter_pov_entries,
    validate_chapter_pov_entries,
)
from .chunking import (
    ChunkingConfig,
    build_rolling_context,
    split_chapter_segments_into_chunks,
    split_text_into_chunks,
)
```

- [ ] **Step 4: Add pass2 chunk builder helper**

Add this function above `run_pass2()`:

```python
def _print_chapter_pov_warnings(warnings: List[str]) -> None:
    for warning in warnings:
        print(f"⚠️ {warning}")


def _build_pass2_chunks(
    text_segment: str,
    chunking_config: ChunkingConfig,
    chapter_pov_path: Optional[str],
    known_character_names,
):
    if not chapter_pov_path:
        return split_text_into_chunks(text_segment, chunking_config)

    entries, load_warnings = load_chapter_pov_entries(chapter_pov_path)
    _print_chapter_pov_warnings(load_warnings)
    validation_warnings = validate_chapter_pov_entries(entries, set(known_character_names))
    _print_chapter_pov_warnings(validation_warnings)

    if not entries:
        return split_text_into_chunks(text_segment, chunking_config)

    matches, match_warnings = find_chapter_title_matches(text_segment, entries)
    _print_chapter_pov_warnings(match_warnings)

    if not matches:
        return split_text_into_chunks(text_segment, chunking_config)

    segments = build_chapter_segments(text_segment, matches)
    for segment in segments:
        if not segment.pov_speaker:
            print("⚠️ chapter segment has no POV")
    return split_chapter_segments_into_chunks(segments, chunking_config)
```

- [ ] **Step 5: Update `run_pass2()` signature and chunk creation**

Update `run_pass2()` signature:

```python
def run_pass2(
    text_segment,
    char_manager,
    chunking_config: ChunkingConfig = None,
    cache_dir: str = "data/llm_cache",
    pass2_workers: int = 4,
    pass2_custom_prompt: str = "",
    chapter_pov_path: Optional[str] = None,
):
```

Replace:

```python
    chunks = split_text_into_chunks(text_segment, chunking_config)
```

with:

```python
    chunks = _build_pass2_chunks(
        text_segment=text_segment,
        chunking_config=chunking_config,
        chapter_pov_path=chapter_pov_path,
        known_character_names=char_manager.characters.keys(),
    )
```

- [ ] **Step 6: Pass chunk metadata into prompt formatting**

Inside `_process_chunk()`, update the `PASS2_PROMPT_TEMPLATE.format()` call:

```python
                "content": PASS2_PROMPT_TEMPLATE.format(
                    available_characters_str=updated_known,
                    previous_chunk_context_str=previous_context_map[chunk.index],
                    chunk_index=chunk.index,
                    total_chunks=len(chunks),
                    chapter_title=chunk.chapter_title or "未提供",
                    pov_speaker=chunk.pov_speaker or "未提供",
                    project_pass2_prompt=_normalize_custom_prompt(pass2_custom_prompt),
                ),
```

- [ ] **Step 7: Run tests and verify pass**

Run:

```bash
pytest tests/test_pipeline_chapter_pov.py tests/test_chunking.py tests/test_chapter_pov.py -v
```

Expected: all chapter POV related tests pass.

- [ ] **Step 8: Commit**

Run:

```bash
git add gnosis/pipeline.py tests/test_pipeline_chapter_pov.py
git commit -m "feat: use chapter POV metadata in pass2"
```

---

### Task 7: Wire CLI Project Paths

**Files:**
- Modify: `main.py`
- Modify: `tests/test_pipeline_chapter_pov.py`

- [ ] **Step 1: Add a focused CLI wiring assertion by static test**

Append this test to `tests/test_pipeline_chapter_pov.py`:

```python
from pathlib import Path


def test_main_wires_chapter_pov_path_to_passes():
    source = Path("main.py").read_text(encoding="utf-8")

    assert 'chapter_pov_path = os.path.join(project_root, "chapter_pov.json")' in source
    assert "chapter_pov_path=chapter_pov_path" in source
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_pipeline_chapter_pov.py::test_main_wires_chapter_pov_path_to_passes -v
```

Expected: fail because `main.py` does not define or pass `chapter_pov_path`.

- [ ] **Step 3: Update `main.py` path definitions**

Near existing project artifact paths, add:

```python
    chapter_pov_path = os.path.join(project_root, "chapter_pov.json")
```

The block should become:

```python
    script_path = os.path.join(project_root, "script.json")
    audio_dir = os.path.join(project_root, "output_audio")
    character_db_path = os.path.join(project_root, "character_db.json")
    chapter_pov_path = os.path.join(project_root, "chapter_pov.json")
    llm_cache_dir = resolve_project_path(project_root, args.llm_cache_dir)
```

- [ ] **Step 4: Pass `chapter_pov_path` into pass1 and pass2**

Update the `run_pass1()` call:

```python
        run_pass1(
            text,
            char_manager,
            pass1_chunking_config,
            cache_dir=llm_cache_dir,
            pass1_custom_prompt=project_prompts.pass1_prompt,
            chapter_pov_path=chapter_pov_path,
        )  # 内部会自动 save_db
```

Update the `run_pass2()` call:

```python
        script_data = run_pass2(
            text,
            char_manager,
            pass2_chunking_config,
            cache_dir=llm_cache_dir,
            pass2_workers=args.pass2_workers,
            pass2_custom_prompt=project_prompts.pass2_prompt,
            chapter_pov_path=chapter_pov_path,
        )
```

- [ ] **Step 5: Run test and verify pass**

Run:

```bash
pytest tests/test_pipeline_chapter_pov.py::test_main_wires_chapter_pov_path_to_passes -v
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add main.py tests/test_pipeline_chapter_pov.py
git commit -m "feat: wire chapter POV artifact through CLI"
```

---

### Task 8: Full Regression Test Pass and Manual Smoke Check

**Files:**
- No new files.
- Verify: `gnosis/models.py`, `gnosis/chapter_pov.py`, `gnosis/chunking.py`, `gnosis/llm_director.py`, `gnosis/pipeline.py`, `main.py`, `tests/test_chapter_pov.py`, `tests/test_chunking.py`, `tests/test_pipeline_chapter_pov.py`

- [ ] **Step 1: Run focused chapter POV tests**

Run:

```bash
pytest tests/test_chapter_pov.py tests/test_chunking.py tests/test_pipeline_chapter_pov.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run existing unit tests likely affected by model or pipeline changes**

Run:

```bash
pytest tests/test_proofread_web.py tests/test_tts_utils.py tests/test_sovits_engine.py -v
```

Expected: all tests pass. These tests verify existing script and character payload consumers still tolerate unchanged artifact shapes.

- [ ] **Step 3: Run complete test suite**

Run:

```bash
pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 4: Run static smoke check for pass2 prompt formatting**

Run:

```bash
python - <<'PY'
from gnosis.llm_director import PASS2_PROMPT_TEMPLATE

prompt = PASS2_PROMPT_TEMPLATE.format(
    available_characters_str="- 绫濑沙季 (female)",
    previous_chunk_context_str="无",
    chunk_index=1,
    total_chunks=1,
    chapter_title="第四卷 9月3日（星期四）绫濑沙季",
    pov_speaker="绫濑沙季",
    project_pass2_prompt="无",
)
assert "章节标题：第四卷 9月3日（星期四）绫濑沙季" in prompt
assert "第一人称视角：绫濑沙季" in prompt
print("pass2 prompt smoke ok")
PY
```

Expected output:

```text
pass2 prompt smoke ok
```

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git diff --stat HEAD
```

Expected: no unstaged changes if each task was committed. If the executor intentionally batched commits, the diff should only include the files listed in this plan.

- [ ] **Step 6: Commit any verification-only corrections**

If Step 1 through Step 4 revealed small mistakes and they were corrected, run:

```bash
git add gnosis/models.py gnosis/chapter_pov.py gnosis/chunking.py gnosis/llm_director.py gnosis/pipeline.py main.py tests/test_chapter_pov.py tests/test_chunking.py tests/test_pipeline_chapter_pov.py
git commit -m "test: cover chapter POV pipeline"
```

Expected: commit succeeds only if there are intentional changes from verification corrections.

---

## Self-Review

Spec coverage:

- `chapter_pov.json` artifact: Tasks 2, 5, and 7.
- Pass1 extracts characters plus chapter POV: Tasks 1, 4, and 5.
- Manual review preservation: Task 2.
- Pass2 chapter title string matching: Task 2.
- Chapter-first chunking: Task 3.
- Prompt metadata injection: Tasks 4 and 6.
- Warning-only behavior: Tasks 2 and 6.
- CLI project wiring: Task 7.
- Tests for every required behavior: Tasks 1 through 8.

Placeholder scan:

- Every function name used by later tasks is declared in an earlier task.
- Every model name used by later tasks is declared in an earlier task.
- Every planned behavior has an owning task.
- No UI, TTS, merge, QA, or proofread expansion.

Type consistency:

- `ChapterPovEntry.chapter_title` and `ChapterPovEntry.pov_speaker` are used consistently in models, persistence, segmentation, chunk metadata, and prompt formatting.
- `ChapterSegment` lives in `gnosis.chapter_pov` and is consumed by `gnosis.chunking`.
- `TextChunk.chapter_title` and `TextChunk.pov_speaker` are optional strings throughout the plan.
