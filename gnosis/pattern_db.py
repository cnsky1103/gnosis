"""
Failure pattern database for TTS quality learning.

Aggregates qa.json data across projects to track failure rates by:
- voice_seed
- text_length_bucket (short/medium/long)
- punctuation patterns (ellipsis, brackets, etc.)
- voice archetype tag

Stored at data/pattern_db.json, checked into repo for portfolio visibility.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from gnosis.qa import _text_length_bucket, _has_special_punctuation


def _pattern_key(voice_seed: str, bucket: str, has_punct: bool) -> str:
    return f"{voice_seed}|{bucket}|{has_punct}"


def _file_fingerprint(path: str) -> str:
    """Hash of file path + mtime for deduplication."""
    import hashlib
    stat = os.stat(path)
    key = f"{os.path.abspath(path)}:{stat.st_mtime_ns}"
    return hashlib.md5(key.encode()).hexdigest()


def aggregate(qa_json_path: str, pattern_db_path: str) -> Dict:
    """
    Read qa.json, merge stats into pattern_db.json.
    Creates pattern_db.json if it doesn't exist.
    Skips if this exact qa.json (same path + mtime) was already aggregated.
    Returns the updated pattern DB.
    """
    with open(qa_json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Load existing DB or create new
    db = _load_or_create(pattern_db_path)

    # Deduplication: skip if already aggregated
    fingerprint = _file_fingerprint(qa_json_path)
    aggregated = set(db.get("aggregated_files", []))
    if fingerprint in aggregated:
        return db
    aggregated.add(fingerprint)
    db["aggregated_files"] = list(aggregated)

    # Count per pattern
    counts = defaultdict(lambda: {"total": 0, "failures": 0})

    for line in qa_data.get("lines", []):
        if line.get("script_char_len", 0) == 0:
            continue  # skip punctuation-only lines

        voice_seed = line.get("voice_seed", "unknown")
        bucket = _text_length_bucket(line.get("text", ""))
        has_punct = _has_special_punctuation(line.get("text", ""))
        key = _pattern_key(voice_seed, bucket, has_punct)

        counts[key]["total"] += 1
        if line.get("status") == "human_review":
            counts[key]["failures"] += 1

    # Merge into DB
    existing_patterns = {
        _pattern_key(p["voice_seed"], p["text_length_bucket"], p["has_special_punctuation"]): p
        for p in db.get("patterns", [])
    }

    for key, stats in counts.items():
        voice_seed, bucket, has_punct_str = key.split("|")
        has_punct = has_punct_str == "True"

        if key in existing_patterns:
            p = existing_patterns[key]
            p["sample_count"] += stats["total"]
            p["failure_count"] = p.get("failure_count", 0) + stats["failures"]
            p["failure_rate"] = round(
                p["failure_count"] / p["sample_count"], 4
            ) if p["sample_count"] > 0 else 0
        else:
            existing_patterns[key] = {
                "voice_seed": voice_seed,
                "text_length_bucket": bucket,
                "has_special_punctuation": has_punct,
                "failure_rate": round(
                    stats["failures"] / stats["total"], 4
                ) if stats["total"] > 0 else 0,
                "failure_count": stats["failures"],
                "sample_count": stats["total"],
            }

    db["patterns"] = list(existing_patterns.values())
    db["total_lines_processed"] = db.get("total_lines_processed", 0) + sum(
        s["total"] for s in counts.values()
    )
    db["updated"] = datetime.now(timezone.utc).isoformat()

    # Write back
    os.makedirs(os.path.dirname(pattern_db_path) or ".", exist_ok=True)
    with open(pattern_db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    return db


def query_risk(pattern_db_path: str, voice_seed: str, text: str) -> float:
    """
    Query the pattern DB for a risk score (0.0 = no risk, 1.0 = very risky).
    Returns 0.0 if no data available.
    """
    if not os.path.exists(pattern_db_path):
        return 0.0

    with open(pattern_db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    bucket = _text_length_bucket(text)
    has_punct = _has_special_punctuation(text)
    key = _pattern_key(voice_seed, bucket, has_punct)

    for p in db.get("patterns", []):
        pkey = _pattern_key(
            p["voice_seed"], p["text_length_bucket"], p["has_special_punctuation"]
        )
        if pkey == key and p.get("sample_count", 0) >= 10:
            return p.get("failure_rate", 0.0)

    return 0.0


def _load_or_create(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": 1,
        "updated": datetime.now(timezone.utc).isoformat(),
        "total_lines_processed": 0,
        "patterns": [],
    }
