"""
Producer-consumer QA pipeline for TTS audio verification.

Architecture:
  ┌──────────────┐     ┌──────────┐     ┌──────────────┐
  │  TTS Job     │────>│ Workers  │────>│ Verify Queue │
  │  Queue       │<────│ (x8)     │     │              │
  └──────────────┘     └──────────┘     └──────┬───────┘
        ▲  retry                               │
        └──────────────────────────────────────┘
                    ASR Verifier (x1)

Termination: in-flight counter (+1 enqueue, -1 on PASS or HUMAN_REVIEW)
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from gnosis.utils import clean_text, is_punctuation_only_text

# Punctuation patterns to strip on retry attempt 2
STRIP_PUNCTUATION_RE = re.compile(
    r'[\.]{2,6}|——|（|）|【|】|「|」|『|』|《|》|〈|〉|…+'
)

MIN_RATIO = 0.8
MAX_RATIO = 1.2


def _text_length_bucket(text: str) -> str:
    n = len(clean_text(text))
    if n < 20:
        return "short"
    elif n <= 50:
        return "medium"
    return "long"


def _has_special_punctuation(text: str) -> bool:
    return bool(STRIP_PUNCTUATION_RE.search(text))


def _strip_punctuation_for_tts(text: str) -> str:
    return STRIP_PUNCTUATION_RE.sub("", text).strip()


class QAResult:
    __slots__ = (
        "index", "speaker", "text", "status", "asr_char_len",
        "script_char_len", "ratio", "retries", "retry_reasons",
        "audio_path", "voice_seed", "voice_tag", "punctuation_stripped",
    )

    def __init__(self, index: int, speaker: str, text: str, voice_seed: str,
                 voice_tag: str, audio_path: str):
        self.index = index
        self.speaker = speaker
        self.text = text
        self.status = "pass"
        self.asr_char_len = 0
        self.script_char_len = 0
        self.ratio = 0.0
        self.retries = 0
        self.retry_reasons: List[str] = []
        self.audio_path = audio_path
        self.voice_seed = voice_seed
        self.voice_tag = voice_tag
        self.punctuation_stripped = False

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "speaker": self.speaker,
            "text": self.text,
            "status": self.status,
            "asr_char_len": self.asr_char_len,
            "script_char_len": self.script_char_len,
            "ratio": round(self.ratio, 4),
            "retries": self.retries,
            "retry_reasons": self.retry_reasons,
            "audio_path": self.audio_path,
            "voice_seed": self.voice_seed,
            "voice_tag": self.voice_tag,
            "punctuation_stripped": self.punctuation_stripped,
        }


class QAPipeline:
    """
    Producer-consumer pipeline for TTS generation + ASR verification.

    8 TTS workers pull jobs from a queue, generate audio, push results
    to a verify queue. 1 ASR verifier checks char length ratios and
    re-enqueues failures for retry (max 2 retries).
    """

    def __init__(
        self,
        tts_engine,
        audio_dir: str,
        num_workers: int = 8,
        whisper_model_size: str = "tiny",
        progress_callback=None,
    ):
        self.tts_engine = tts_engine
        self.audio_dir = audio_dir
        self.num_workers = num_workers
        self.whisper_model_size = whisper_model_size
        self.progress_callback = progress_callback

        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.verify_queue: asyncio.Queue = asyncio.Queue()
        self.results: List[QAResult] = []
        self.in_flight = 0
        self.done_event = asyncio.Event()

        # Counters for progress
        self.total_jobs = 0
        self.pass_count = 0
        self.retry_count = 0
        self.human_review_count = 0
        self.completed_count = 0

        self._whisper_model = None
        self._whisper_available = True

    def _load_whisper(self):
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel(
                self.whisper_model_size, device="cpu", compute_type="int8"
            )
            return self._whisper_model
        except Exception as exc:
            print(f"⚠️ Whisper 加载失败: {exc}")
            self._whisper_available = False
            return None

    def _transcribe(self, audio_path: str) -> str:
        model = self._load_whisper()
        if model is None:
            return ""
        try:
            segments, _info = model.transcribe(audio_path, language="zh")
            return "".join([s.text for s in segments])
        except Exception:
            return ""

    async def run(
        self,
        jobs: List[Dict[str, Any]],
        speaker_to_voice: Dict[str, str],
        characters: List[Dict],
    ) -> Dict:
        """
        Run the full pipeline. Returns qa.json-compatible dict.

        jobs: list of (index, line_dict) from filter_script_jobs_by_character
        speaker_to_voice: {speaker_name: voice_seed_id}
        characters: list of character dicts from script.json
        """
        char_to_tag = {}
        for c in characters:
            char_to_tag[c.get("name", "")] = c.get("voice_archetype", "")

        # Separate auto-pass lines and TTS-needing lines
        for idx, line in jobs:
            text = line.get("text", "")
            speaker = line.get("speaker", "")
            voice_seed = speaker_to_voice.get(speaker, "logos")
            voice_tag = char_to_tag.get(speaker, "")
            audio_path = os.path.join(self.audio_dir, f"{idx:04d}.wav")

            result = QAResult(
                index=idx, speaker=speaker, text=text,
                voice_seed=voice_seed, voice_tag=voice_tag,
                audio_path=audio_path,
            )

            if is_punctuation_only_text(text) or len(clean_text(text)) == 0:
                result.status = "pass"
                result.script_char_len = 0
                self.results.append(result)
                self.total_jobs += 1
                self.completed_count += 1
                self._notify_progress()
                continue

            if os.path.exists(audio_path):
                # Existing WAV: verify only, don't regenerate
                self._enqueue_verify_only(result)
                continue

            # Enqueue for TTS generation
            job = {
                "index": idx,
                "text": text,
                "speaker": speaker,
                "voice_seed": voice_seed,
                "attempt": 0,
                "strategy": "default",
                "original_text": text,
                "result": result,
            }
            self.tts_queue.put_nowait(job)
            self.in_flight += 1
            self.total_jobs += 1

        if self.in_flight == 0:
            self.done_event.set()

        # Start workers and verifier
        workers = [
            asyncio.create_task(self._tts_worker(i))
            for i in range(min(self.num_workers, max(1, self.in_flight)))
        ]
        verifier = asyncio.create_task(self._verifier_loop())

        await self.done_event.wait()

        # Signal workers and verifier to stop
        for _ in workers:
            self.tts_queue.put_nowait(None)  # sentinel
        self.verify_queue.put_nowait(None)  # sentinel

        await asyncio.gather(*workers, verifier)

        return self._build_qa_report()

    def _enqueue_verify_only(self, result: QAResult):
        """For existing WAV files: enqueue directly to verify queue."""
        self.in_flight += 1
        self.total_jobs += 1
        self.verify_queue.put_nowait({
            "result": result,
            "verify_only": True,
        })

    async def _tts_worker(self, worker_id: int):
        while True:
            job = await self.tts_queue.get()
            if job is None:
                self.tts_queue.task_done()
                break

            try:
                await self._process_tts_job(job)
            except Exception as exc:
                # Catch-all: push error result to verify queue
                result = job["result"]
                self.verify_queue.put_nowait({
                    "result": result,
                    "error": str(exc),
                    "attempt": job["attempt"],
                })
            finally:
                self.tts_queue.task_done()

    async def _process_tts_job(self, job: Dict):
        result = job["result"]
        text = job["text"]
        voice_seed = job["voice_seed"]
        attempt = job["attempt"]
        strategy = job["strategy"]

        # Apply retry strategy
        tts_text = text
        if strategy == "strip_punctuation":
            tts_text = _strip_punctuation_for_tts(text)
            if not tts_text.strip():
                tts_text = text  # fallback if stripping removed everything
            else:
                result.punctuation_stripped = True

        audio_path = result.audio_path

        # Delete old WAV if retrying
        if attempt > 0 and os.path.exists(audio_path):
            os.remove(audio_path)

        try:
            ok = await asyncio.wait_for(
                self.tts_engine.generate_line(
                    text=tts_text,
                    output_path=audio_path,
                    voice_id=voice_seed,
                ),
                timeout=300,  # 5 min timeout for zero-shot TTS
            )
        except asyncio.TimeoutError:
            ok = False
            if os.path.exists(audio_path):
                os.remove(audio_path)

        self.verify_queue.put_nowait({
            "result": result,
            "ok": ok,
            "attempt": attempt,
        })

    async def _verifier_loop(self):
        while True:
            item = await self.verify_queue.get()
            if item is None:
                self.verify_queue.task_done()
                break

            try:
                await self._verify_item(item)
            except Exception as exc:
                # If verification itself crashes, mark as human_review
                result = item.get("result")
                if result:
                    result.status = "human_review"
                    result.retry_reasons.append(f"verify_crash: {exc}")
                    self.results.append(result)
                    self.human_review_count += 1
                    self.completed_count += 1
                    self._notify_progress()
                    self.in_flight -= 1
                    if self.in_flight == 0:
                        self.done_event.set()
            finally:
                self.verify_queue.task_done()

    async def _verify_item(self, item: Dict):
        result: QAResult = item["result"]
        verify_only = item.get("verify_only", False)
        error = item.get("error")
        attempt = item.get("attempt", 0)

        # Handle TTS errors
        if error:
            if attempt < 2:
                self._retry(result, attempt, f"tts_error: {error}")
                return
            result.status = "human_review"
            result.retries = attempt
            result.retry_reasons.append(f"tts_error: {error}")
            self.results.append(result)
            self.human_review_count += 1
            self.completed_count += 1
            self._notify_progress()
            self.in_flight -= 1
            if self.in_flight == 0:
                self.done_event.set()
            return

        # Handle TTS failure (returned False)
        if not verify_only and not item.get("ok", True):
            if attempt < 2:
                self._retry(result, attempt, "tts_returned_false")
                return
            result.status = "human_review"
            result.retries = attempt
            result.retry_reasons.append("tts_returned_false")
            self.results.append(result)
            self.human_review_count += 1
            self.completed_count += 1
            self._notify_progress()
            self.in_flight -= 1
            if self.in_flight == 0:
                self.done_event.set()
            return

        # ASR verification
        audio_path = result.audio_path
        if not os.path.exists(audio_path):
            if verify_only:
                result.status = "human_review"
                result.retry_reasons.append("audio_missing")
                self.results.append(result)
                self.human_review_count += 1
                self.completed_count += 1
                self._notify_progress()
                self.in_flight -= 1
                if self.in_flight == 0:
                    self.done_event.set()
                return
            if attempt < 2:
                self._retry(result, attempt, "audio_not_written")
                return
            result.status = "human_review"
            result.retries = attempt
            result.retry_reasons.append("audio_not_written")
            self.results.append(result)
            self.human_review_count += 1
            self.completed_count += 1
            self._notify_progress()
            self.in_flight -= 1
            if self.in_flight == 0:
                self.done_event.set()
            return

        if not self._whisper_available:
            result.status = "human_review"
            result.retry_reasons.append("asr_unavailable")
            self.results.append(result)
            self.human_review_count += 1
            self.completed_count += 1
            self._notify_progress()
            self.in_flight -= 1
            if self.in_flight == 0:
                self.done_event.set()
            return

        # Run ASR in thread (blocking call)
        asr_text = await asyncio.to_thread(self._transcribe, audio_path)
        asr_cleaned = clean_text(asr_text)
        script_cleaned = clean_text(result.text)

        result.asr_char_len = len(asr_cleaned)
        result.script_char_len = len(script_cleaned)

        if result.script_char_len == 0:
            result.ratio = float(result.asr_char_len) if result.asr_char_len > 0 else 1.0
        else:
            result.ratio = result.asr_char_len / result.script_char_len

        # Classify
        if MIN_RATIO <= result.ratio <= MAX_RATIO:
            if attempt > 0:
                result.status = "auto_retried"
                self.retry_count += 1
            else:
                result.status = "pass"
                self.pass_count += 1
            result.retries = attempt
            self.results.append(result)
            self.completed_count += 1
            self._notify_progress()
            self.in_flight -= 1
            if self.in_flight == 0:
                self.done_event.set()
        elif attempt < 2:
            reason = f"ratio_{result.ratio:.3f}"
            if not asr_cleaned:
                reason = "asr_empty"
            self._retry(result, attempt, reason)
        else:
            result.status = "human_review"
            result.retries = attempt
            reason = f"ratio_{result.ratio:.3f}"
            if not asr_cleaned:
                reason = "asr_empty"
            result.retry_reasons.append(reason)
            self.results.append(result)
            self.human_review_count += 1
            self.completed_count += 1
            self._notify_progress()
            self.in_flight -= 1
            if self.in_flight == 0:
                self.done_event.set()

    def _retry(self, result: QAResult, current_attempt: int, reason: str):
        next_attempt = current_attempt + 1
        result.retry_reasons.append(reason)

        if next_attempt == 1:
            strategy = "default"  # just regenerate
        else:
            strategy = "strip_punctuation"

        job = {
            "index": result.index,
            "text": result.text,
            "speaker": result.speaker,
            "voice_seed": result.voice_seed,
            "attempt": next_attempt,
            "strategy": strategy,
            "original_text": result.text,
            "result": result,
        }
        self.tts_queue.put_nowait(job)
        # in_flight stays the same (not decremented, not incremented)

    def _notify_progress(self):
        if self.progress_callback:
            self.progress_callback(
                completed=self.completed_count,
                total=self.total_jobs,
                passed=self.pass_count,
                retried=self.retry_count,
                human_review=self.human_review_count,
            )

    def _build_qa_report(self) -> Dict:
        sorted_results = sorted(self.results, key=lambda r: r.index)
        total = len(sorted_results)
        pass_count = sum(1 for r in sorted_results if r.status == "pass")
        retried_count = sum(1 for r in sorted_results if r.status == "auto_retried")
        review_count = sum(1 for r in sorted_results if r.status == "human_review")

        return {
            "summary": {
                "total": total,
                "pass": pass_count,
                "auto_retried": retried_count,
                "human_review": review_count,
                "pass_rate": round((pass_count + retried_count) / total, 4) if total > 0 else 0,
            },
            "lines": [r.to_dict() for r in sorted_results],
        }


async def run_verify_only(
    script_path: str,
    audio_dir: str,
    whisper_model_size: str = "tiny",
    progress_callback=None,
) -> Dict:
    """Standalone verify: run ASR on existing audio, produce qa.json. No TTS."""
    from gnosis.utils import parse_script_payload, collect_sorted_segments

    with open(script_path, "r", encoding="utf-8") as f:
        script_payload = json.load(f)
    characters, script_list = parse_script_payload(script_payload)

    char_to_tag = {}
    for c in characters:
        char_to_tag[c.get("name", "")] = c.get("voice_archetype", "")

    # Load whisper
    whisper_model = None
    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
    except Exception as exc:
        print(f"⚠️ Whisper 加载失败: {exc}")

    results = []
    total = len(script_list)

    for i, line in enumerate(script_list):
        text = line.get("text", "")
        speaker = line.get("speaker", "")
        voice_tag = char_to_tag.get(speaker, "")
        audio_path = os.path.join(audio_dir, f"{i:04d}.wav")

        result = {
            "index": i,
            "speaker": speaker,
            "text": text,
            "status": "pass",
            "asr_char_len": 0,
            "script_char_len": 0,
            "ratio": 0.0,
            "retries": 0,
            "retry_reasons": [],
            "audio_path": audio_path,
            "voice_seed": "",
            "voice_tag": voice_tag,
            "punctuation_stripped": False,
        }

        if is_punctuation_only_text(text) or len(clean_text(text)) == 0:
            result["status"] = "pass"
            results.append(result)
            if progress_callback:
                progress_callback(completed=i + 1, total=total, passed=0, retried=0, human_review=0)
            continue

        if not os.path.exists(audio_path):
            result["status"] = "human_review"
            result["retry_reasons"] = ["audio_missing"]
            results.append(result)
            if progress_callback:
                progress_callback(completed=i + 1, total=total, passed=0, retried=0, human_review=0)
            continue

        if whisper_model is None:
            result["status"] = "human_review"
            result["retry_reasons"] = ["asr_unavailable"]
            results.append(result)
            if progress_callback:
                progress_callback(completed=i + 1, total=total, passed=0, retried=0, human_review=0)
            continue

        try:
            segments, _info = whisper_model.transcribe(audio_path, language="zh")
            asr_text = "".join([s.text for s in segments])
        except Exception:
            asr_text = ""

        asr_cleaned = clean_text(asr_text)
        script_cleaned = clean_text(text)
        result["asr_char_len"] = len(asr_cleaned)
        result["script_char_len"] = len(script_cleaned)

        if result["script_char_len"] == 0:
            result["ratio"] = float(result["asr_char_len"]) if result["asr_char_len"] > 0 else 1.0
        else:
            result["ratio"] = round(result["asr_char_len"] / result["script_char_len"], 4)

        if MIN_RATIO <= result["ratio"] <= MAX_RATIO:
            result["status"] = "pass"
        else:
            result["status"] = "human_review"
            reason = f"ratio_{result['ratio']:.3f}"
            if not asr_cleaned:
                reason = "asr_empty"
            result["retry_reasons"] = [reason]

        results.append(result)
        if progress_callback:
            progress_callback(completed=i + 1, total=total, passed=0, retried=0, human_review=0)

    total_r = len(results)
    pass_c = sum(1 for r in results if r["status"] == "pass")
    review_c = sum(1 for r in results if r["status"] == "human_review")

    return {
        "summary": {
            "total": total_r,
            "pass": pass_c,
            "auto_retried": 0,
            "human_review": review_c,
            "pass_rate": round(pass_c / total_r, 4) if total_r > 0 else 0,
        },
        "lines": results,
    }


def format_qa_report(qa_data: Dict) -> str:
    """Format qa.json data as a human-readable report."""
    summary = qa_data["summary"]
    lines = qa_data["lines"]

    out = []
    out.append("=" * 60)
    out.append("  QA REPORT")
    out.append("=" * 60)
    out.append(f"  Total lines:   {summary['total']}")
    out.append(f"  Pass:          {summary['pass']}")
    out.append(f"  Auto-retried:  {summary['auto_retried']}")
    out.append(f"  Human review:  {summary['human_review']}")
    out.append(f"  Pass rate:     {summary['pass_rate']:.1%}")
    out.append("")

    # Failure breakdown by voice seed
    voice_failures = {}
    for line in lines:
        if line["status"] == "human_review":
            seed = line.get("voice_seed", "unknown")
            voice_failures[seed] = voice_failures.get(seed, 0) + 1

    if voice_failures:
        out.append("  Failures by voice seed:")
        for seed, count in sorted(voice_failures.items(), key=lambda x: -x[1]):
            out.append(f"    {seed}: {count}")
        out.append("")

    # List human_review lines
    review_lines = [l for l in lines if l["status"] == "human_review"]
    if review_lines:
        out.append("  Lines needing review:")
        for l in review_lines[:20]:  # cap at 20
            text_preview = l["text"][:40] + "..." if len(l["text"]) > 40 else l["text"]
            out.append(
                f"    [{l['index']:04d}] {l['speaker']}: {text_preview}"
                f" (ratio={l['ratio']:.3f}, reasons={l['retry_reasons']})"
            )
        if len(review_lines) > 20:
            out.append(f"    ... and {len(review_lines) - 20} more")

    out.append("=" * 60)
    return "\n".join(out)


def export_qa_markdown(qa_data: Dict, project_name: str) -> str:
    """Export qa.json data as a Markdown report for README/portfolio."""
    summary = qa_data["summary"]
    lines = qa_data["lines"]

    out = []
    out.append(f"# QA Report: {project_name}")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|--------|-------|")
    out.append(f"| Total lines | {summary['total']} |")
    out.append(f"| Pass | {summary['pass']} |")
    out.append(f"| Auto-retried | {summary['auto_retried']} |")
    out.append(f"| Human review | {summary['human_review']} |")
    out.append(f"| Pass rate | {summary['pass_rate']:.1%} |")
    out.append("")

    # Voice seed failure rates
    voice_stats = {}
    for line in lines:
        seed = line.get("voice_seed", "unknown")
        if seed not in voice_stats:
            voice_stats[seed] = {"total": 0, "failures": 0}
        voice_stats[seed]["total"] += 1
        if line["status"] == "human_review":
            voice_stats[seed]["failures"] += 1

    if voice_stats:
        out.append("## Voice Seed Performance")
        out.append("")
        out.append("| Voice Seed | Total Lines | Failures | Failure Rate |")
        out.append("|------------|-------------|----------|-------------|")
        for seed, stats in sorted(voice_stats.items(), key=lambda x: -x[1]["failures"]):
            rate = stats["failures"] / stats["total"] if stats["total"] > 0 else 0
            out.append(f"| {seed} | {stats['total']} | {stats['failures']} | {rate:.1%} |")
        out.append("")

    # Text length analysis
    bucket_stats = {"short": {"total": 0, "fail": 0}, "medium": {"total": 0, "fail": 0}, "long": {"total": 0, "fail": 0}}
    for line in lines:
        bucket = _text_length_bucket(line["text"])
        bucket_stats[bucket]["total"] += 1
        if line["status"] == "human_review":
            bucket_stats[bucket]["fail"] += 1

    out.append("## Failure Rate by Text Length")
    out.append("")
    out.append("| Bucket | Total | Failures | Rate |")
    out.append("|--------|-------|----------|------|")
    for bucket in ["short", "medium", "long"]:
        s = bucket_stats[bucket]
        rate = s["fail"] / s["total"] if s["total"] > 0 else 0
        out.append(f"| {bucket} (<20 / 20-50 / >50 chars) | {s['total']} | {s['fail']} | {rate:.1%} |")

    out.append("")
    out.append(f"*Generated by Gnosis QA Pipeline on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*")

    return "\n".join(out)
