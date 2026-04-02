# Gnosis

A production-grade audiobook pipeline that converts Chinese light novels into multi-character audio dramas with automated quality assurance.

Built as a solo project. 15 novels produced, 30,000+ replays on Bilibili.

## What it does

Gnosis takes a novel text file and produces a finished audiobook with synchronized subtitles:

```
Novel text  →  Character extraction (LLM)  →  Script generation (LLM)
    →  TTS audio (CosyVoice)  →  QA verification (ASR)  →  Audio merge  →  SRT subtitles
```

Each step is a CLI command. Run them individually or `full` for end-to-end.

## Architecture

The QA pipeline is a producer-consumer system that verifies every audio segment automatically:

```
┌─────────────────────────────────────────────────────────┐
│                     main.py                              │
│  python main.py tts --project <name>                     │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │     TTS + Verify Pipeline      │
          │                                │
          │  ┌──────────────────────────┐  │
          │  │    asyncio.Queue         │  │
          │  │    (Job Queue)           │  │
          │  │                          │  │
          │  │  Initial: all lines      │  │
          │  │  Retries: re-enqueued    │  │
          │  └─────┬──────────┬────────┘  │
          │        │          │           │
          │   ┌────┴───┐ ┌───┴────┐      │
          │   │Worker 1│ │Worker N│ x8   │
          │   │CosyVoice│ │CosyVoice│     │
          │   └────┬───┘ └───┬────┘      │
          │        │          │           │
          │   ┌────┴──────────┴────┐      │
          │   │   ASR Verifier     │      │
          │   │                    │      │
          │   │ Whisper → char len │      │
          │   │ ratio check        │      │
          │   │                    │      │
          │   │ PASS  → done       │      │
          │   │ FAIL  → re-enqueue │──┐   │
          │   │ FINAL → human_review│  │  │
          │   └────────────────────┘  │   │
          │        ▲                  │   │
          │        └──────────────────┘   │
          └───────────────┬───────────────┘
                          │
               qa.json (audit trail)
```

**How verification works:** After generating each audio segment, Whisper (ASR) transcribes it back to text. The system compares the character count of the transcription against the original script. If the ratio is outside 0.8-1.2, the segment is flagged and automatically retried with a different strategy. After 2 retries, it's escalated for manual review.

This catches CosyVoice's occasional sentence-dropping bug without slowing down the pipeline. ASR runs within the TTS wall time (TTS: ~5.5h per novel, ASR: ~30min).

**Why producer-consumer:** TTS workers stay busy generating audio while the verifier checks previous segments. Retries flow back into the same job queue, naturally load-balanced across workers. An in-flight counter tracks completion: increment on enqueue, decrement on final disposition. Pipeline terminates when counter reaches zero.

## Pipeline stages

| Stage | Command | What it does |
|-------|---------|-------------|
| Extract | `python main.py extract --project X` | LLM identifies characters from the novel text |
| Script | `python main.py script --project X` | LLM generates structured dialogue with speaker attribution |
| TTS + QA | `python main.py tts --project X` | CosyVoice generates audio, ASR verifies, auto-retries failures |
| Verify | `python main.py verify --project X` | Re-verify existing audio without regenerating |
| Merge | `python main.py merge --project X` | Sample-precise audio concatenation + normalization |
| Proofread | `python main.py proofread --project X` | Web UI for manual review with QA highlighting |
| QA Report | `python main.py qa-report --project X` | Formatted quality summary |
| QA Export | `python main.py qa-export --project X` | Markdown quality report |

## QA system

The QA system produces `qa.json` for each project with full audit trail:

```json
{
  "summary": {
    "total": 2500,
    "pass": 2380,
    "auto_retried": 95,
    "human_review": 25,
    "pass_rate": 0.99
  },
  "lines": [
    {
      "index": 42,
      "speaker": "Character Name",
      "text": "Original script text",
      "status": "pass",
      "asr_char_len": 9,
      "script_char_len": 8,
      "ratio": 1.125,
      "retries": 0,
      "voice_seed": "duanya",
      "voice_tag": "男-普通"
    }
  ]
}
```

**Three-tier classification:**
- `pass` — ASR char ratio within 0.8-1.2. Audio is good.
- `auto_retried` — Failed initial check, passed after retry. Logged for pattern analysis.
- `human_review` — Failed all retries. Flagged in the proofread web UI.

**Retry strategies:**
1. Regenerate with same parameters (handles transient failures)
2. Strip special punctuation from TTS input (handles punctuation-related drops)

## Pattern database

`data/pattern_db.json` aggregates failure data across projects:

- Which voice seeds fail most often
- Failure rate by text length (short/medium/long)
- Whether special punctuation correlates with failures

This data accumulates over time, making the system smarter with each novel processed.

## Proofread web UI

```
python main.py proofread --project X
```

Opens a web-based editor at `http://127.0.0.1:8765` for manual review:

- Full novel view with speaker attribution
- QA-flagged lines highlighted: red border (human_review), yellow border (retried)
- Audio playback per line
- Character ratio badges
- "Next flagged" button jumps to lines needing attention
- Keyboard shortcuts for speaker assignment

## Tech stack

- **Python 3.10+**, asyncio for concurrency
- **CosyVoice** (zero-shot TTS, 31 voice seeds)
- **Faster-Whisper** (ASR verification, tiny model, int8)
- **DeepSeek** (LLM for character extraction + script generation)
- **Rich** (progress bars)
- **NumPy + SoundFile** (sample-precise audio processing)
- **Pydantic** (data validation)

## Project structure

```
gnosis/
├── qa.py              # Producer-consumer QA pipeline
├── pattern_db.py      # Failure pattern learning
├── pipeline.py        # LLM orchestration (Pass 1 + Pass 2)
├── tts/
│   ├── cosy_voice_engine.py  # CosyVoice TTS integration
│   ├── tts_engine.py         # Base TTS interface
│   └── tts_utils.py          # Audio utilities
├── merge_audio.py     # Sample-precise concatenation
├── srt.py             # Subtitle generation
├── proofread_web.py   # Web UI server
├── web/proofread/     # Frontend (HTML/JS/CSS)
├── config.py          # Voice seed mappings (31 voices)
└── utils.py           # Text cleaning, file utilities

data/
├── projects/          # Per-novel project directories
│   ├── <novel>/
│   │   ├── script.json       # Generated dialogue script
│   │   ├── qa.json           # QA audit trail
│   │   ├── output_audio/     # Generated WAV segments
│   │   └── final_audiobook.wav
│   └── ...
└── pattern_db.json    # Cross-project failure patterns

tests/
└── test_qa.py         # Pipeline tests (mock TTS + mock ASR)
```

## Running tests

```bash
python -m pytest tests/ -v
```

All tests use mock TTS and mock Whisper. No GPU required. Runs in <1 second.
