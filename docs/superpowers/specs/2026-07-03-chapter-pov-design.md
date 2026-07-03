# Chapter POV Extraction and Chapter-First Pass2 Design

## Goal

Prevent pass2 speaker attribution from losing first-person point-of-view context when a chapter is split across chunks.

The system should extract a novel-level chapter POV table during `extract`, let the user manually review it, then make `script` split the novel by reviewed chapter titles before creating pass2 chunks. Every pass2 chunk inside a chapter receives the chapter title and POV speaker as structured context.

## Problem

Today `pass2` chunks the whole input by paragraph and character count. A chapter title such as `第四卷 9月3日（星期四）绫濑沙季` can appear in one chunk while later paragraphs from the same chapter appear in a different chunk. When the later chunk lacks the title, the LLM may assign first-person narration to the wrong protagonist.

Current flow:

```text
input.txt
  -> split_text_into_chunks(full_text, pass2_config)
  -> each chunk gets only previous chunk tail context
  -> pass2 LLM guesses first-person narrator from local evidence
```

Target flow:

```text
extract/pass1
  -> character_db.json
  -> chapter_pov.json

manual review
  -> user edits character_db.json
  -> user edits chapter_pov.json

script/pass2
  -> load chapter_pov.json
  -> find each reviewed chapter title in input.txt
  -> split input into chapter segments
  -> split each chapter segment into pass2 chunks
  -> pass chapter title + POV speaker to every chunk prompt
```

## Scope

In scope:

- Add a new reviewed project artifact: `data/projects/<project>/chapter_pov.json`.
- Extend pass1 output parsing so each extract chunk can return new characters and chapter POV entries.
- Persist deduplicated chapter POV entries during `extract` and `full`.
- Make pass2 split by reviewed chapter titles first, then by existing chunk size rules inside each chapter.
- Inject chapter metadata into the pass2 prompt.
- Warn, but do not fail, when chapter POV data is missing or imperfect.
- Add unit tests for chapter POV models, persistence, chapter title matching, chapter-first chunking, prompt injection, and warning behavior.

Out of scope:

- Building a web UI for reviewing `chapter_pov.json`.
- Automatically repairing missing or incorrect chapter POV data.
- Automatically guessing POV in pass2 when reviewed metadata is missing.
- Changing TTS, audio merge, QA, or proofread behavior beyond consuming the unchanged final `script.json`.
- Migrating existing project artifacts.

## Data Model

Add these Pydantic models in `gnosis/models.py`:

```python
from pydantic import Field


class ChapterPovEntry(BaseModel):
    chapter_title: str
    pov_speaker: str
    evidence: Optional[str] = None


class ChapterPovExtraction(BaseModel):
    chapters: List[ChapterPovEntry]


class CharacterExtraction(BaseModel):
    new_characters: List[CharacterProfile]
    chapters: List[ChapterPovEntry] = Field(default_factory=list)
```

`chapter_title` is the exact title text that pass2 will search for in `input.txt`.

`pov_speaker` is the reviewed actor name that should receive first-person narration for that chapter. It should match a `CharacterProfile.name` after user review, but pass2 only warns if it does not.

`evidence` is optional source text explaining why the LLM chose the POV. It is for human review only.

Persisted file format:

```json
[
  {
    "chapter_title": "第四卷 9月3日（星期四）浅村悠太",
    "pov_speaker": "浅村悠太",
    "evidence": "标题末尾出现浅村悠太"
  },
  {
    "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
    "pov_speaker": "绫濑沙季",
    "evidence": "标题末尾出现绫濑沙季"
  }
]
```

## Artifact Ownership

`character_db.json` remains an array of characters. Its top-level shape must not change, because existing TTS and proofreading code expects that format.

`chapter_pov.json` is a sibling artifact in the project directory:

```text
data/projects/<project>/
  input.txt
  character_db.json
  chapter_pov.json
  script.json
```

The user is expected to manually review both `character_db.json` and `chapter_pov.json` after `extract`.

## Pass1 Extraction

Update `PASS1_PROMPT_TEMPLATE` so the LLM returns both newly found characters and chapter POV entries.

Required output shape:

```json
{
  "new_characters": [
    {
      "name": "...",
      "gender": "male",
      "voice_archetype": "男-普通",
      "description": "..."
    }
  ],
  "chapters": [
    {
      "chapter_title": "...",
      "pov_speaker": "...",
      "evidence": "..."
    }
  ]
}
```

Extraction rules:

- Extract chapter entries when the text contains a chapter heading, date heading, diary heading, or other strong section boundary that implies a POV.
- `chapter_title` must be copied exactly from the input text.
- `pov_speaker` must be the character whose first-person narration controls that chapter.
- If the title directly names the POV character, use that name.
- If the title does not name the POV but the chapter clearly establishes it, use the inferred character and put the reasoning in `evidence`.
- If no POV can be identified, omit that chapter entry instead of guessing.

`run_pass1` should collect `extraction.chapters` from every pass1 chunk and pass them to a new chapter POV manager. It should not require a separate LLM request.

## Chapter POV Persistence

Add a small persistence module at `gnosis/chapter_pov.py`.

Responsibilities:

- Load `chapter_pov.json` from a project path.
- Save reviewed entries as a JSON array.
- Add extracted entries while deduplicating by normalized `chapter_title`.
- Preserve existing user-reviewed entries when extract is re-run.

Deduplication rule:

```text
normalize title = strip leading/trailing whitespace and collapse internal whitespace to one space
```

If an extracted title already exists, keep the existing persisted entry unchanged. This protects user edits.

## Pass2 Chapter Segmentation

Before pass2 chunking, load `chapter_pov.json` and find each `chapter_title` in the current input text.

Matching rules:

- Use exact string search after normalizing line endings to `\n`.
- Search titles in the order they appear in `chapter_pov.json`.
- If a title is not found, warn and skip that title.
- If a title appears more than once, warn and use the first occurrence after the previous matched chapter.
- If titles are out of order, warn and skip the out-of-order title.
- Text before the first matched chapter remains a segment with no chapter POV.
- Each matched chapter segment runs from its title start to the next matched title start.
- Text after the last matched chapter belongs to the last matched chapter.

Data flow:

```text
input text
  |
  v
chapter_pov.json entries
  |
  v
find title offsets
  |
  v
ChapterSegment(title, pov_speaker, text)
  |
  v
split each ChapterSegment.text with existing paragraph chunker
  |
  v
TextChunk(index, text, paragraphs, chapter_title, pov_speaker)
```

The existing chunker should keep its paragraph and character-count behavior inside each chapter. The only behavioral change is that a chunk may not cross a reviewed chapter boundary.

## Pass2 Prompt

Extend `TextChunk` with optional metadata:

```python
chapter_title: Optional[str] = None
pov_speaker: Optional[str] = None
```

Update `PASS2_PROMPT_TEMPLATE` with a chapter metadata section:

```text
# 当前章节信息
章节标题：{chapter_title}
第一人称视角：{pov_speaker}

如果“第一人称视角”非空，当前小说片段中所有非对话旁白、第一人称心理活动、第一人称叙述句的 speaker 必须使用该角色名。
```

If no POV is available for a chunk, pass empty strings or `未提供`, and let the existing prompt behavior apply.

## Warning Policy

Warnings are required but non-blocking. The system must not stop script generation for chapter POV problems.

Warn in these cases:

- `chapter_pov.json` is missing during `script`.
- `chapter_pov.json` is invalid JSON.
- A chapter entry lacks `chapter_title` or `pov_speaker`.
- A `pov_speaker` is not present in `character_db.json`.
- A `chapter_title` cannot be found in `input.txt`.
- A `chapter_title` matches out of order.
- A chapter segment has no POV.

Do not:

- Auto-fill missing POV.
- Reassign POV from nearby chapters.
- Ask pass2 to guess missing POV.
- Fail the command.

The user will manually edit `chapter_pov.json` when warnings reveal missing information.

## Cache Behavior

Pass2 request cache keys already include the full prompt and chunk text. Adding chapter metadata to the prompt naturally invalidates affected cache entries. No explicit cache version bump is required.

Pass1 cache keys should change because the prompt output shape changes. Existing pass1 cache entries may parse without `chapters` due to the default empty list, but they will not produce chapter POV data. The implementation should make this visible by printing a warning after extract if zero chapter entries were produced.

## Testing Requirements

Add tests for:

1. `CharacterExtraction` accepts old responses without `chapters` and new responses with `chapters`.
2. Chapter POV manager preserves user-reviewed entries when extract is re-run.
3. Chapter POV manager deduplicates titles by collapsed whitespace.
4. Chapter title matching creates ordered chapter segments.
5. Missing chapter title produces a warning and does not stop segmentation.
6. Out-of-order title produces a warning and is skipped.
7. Text before the first matched chapter becomes a no-POV segment.
8. Each chapter segment is chunked independently, so pass2 chunks never cross chapter boundaries.
9. Every chunk inside a chapter carries the same `chapter_title` and `pov_speaker`.
10. The pass2 prompt includes chapter title and POV when metadata exists.
11. The pass2 prompt uses empty or `未提供` metadata when no POV exists.
12. A `pov_speaker` absent from `character_db.json` warns but does not stop `script`.

Recommended test files:

```text
tests/test_chapter_pov.py
tests/test_chunking.py
tests/test_pipeline_chapter_pov.py
```

## Acceptance Criteria

- Running `python main.py extract --project 义妹4` writes or updates `data/projects/义妹4/chapter_pov.json`.
- Re-running extract does not overwrite manually edited entries in `chapter_pov.json`.
- Running `python main.py script --project 义妹4` reads `chapter_pov.json` and prints warnings for review issues.
- Pass2 chunks are created inside chapter boundaries, not across them.
- A chunk from a `绫濑沙季` chapter receives `pov_speaker = "绫濑沙季"` even if the chunk text does not contain the chapter title.
- Missing or imperfect chapter POV data never blocks script generation.
- Existing projects without `chapter_pov.json` still run `script` with warning-only behavior.
- Existing `character_db.json` and `script.json` shapes remain compatible with TTS, proofread, QA, and merge steps.

## Implementation Notes

Keep the change focused:

- Do not rewrite the whole pipeline.
- Do not introduce a database or global project metadata format.
- Do not add a reviewer UI.
- Prefer small helpers with pure functions for segmentation and warning collection.

The likely file responsibilities are:

```text
gnosis/models.py
  Add ChapterPovEntry and extend CharacterExtraction.

gnosis/chapter_pov.py
  Load/save/deduplicate reviewed chapter POV entries.
  Match reviewed titles against input text.
  Build chapter segments and warnings.

gnosis/chunking.py
  Add optional chapter metadata to TextChunk.
  Add a helper that chunks ChapterSegment objects with the existing paragraph chunker.

gnosis/llm_director.py
  Update pass1 and pass2 prompt templates.

gnosis/pipeline.py
  Wire pass1 chapter extraction into persistence.
  Wire pass2 chapter-first chunking and prompt metadata.

main.py
  Define chapter_pov_path and pass it into pass1/pass2.

tests/
  Add focused unit tests for models, persistence, segmentation, chunk metadata, prompt metadata, and warning-only behavior.
```

## Open Decisions

No open product decisions remain for this feature.

Implementation may still choose exact helper names and warning text, as long as the behavior and artifact shapes above are preserved.
