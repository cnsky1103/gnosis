"""
Tests for the QA producer-consumer pipeline.
Uses mock TTS engine and mock Whisper — no GPU, runs in seconds.
"""

import asyncio
import json
import os
import tempfile

import pytest

from gnosis.qa import QAPipeline, QAResult, MIN_RATIO, MAX_RATIO, run_verify_only


class MockTTSEngine:
    """Mock TTS engine that writes a fake WAV file."""

    def __init__(self, fail_indices=None, permanent_fail_indices=None):
        self.fail_indices = set(fail_indices or [])
        self.permanent_fail_indices = set(permanent_fail_indices or [])
        self.call_count = {}

    async def generate_line(self, text, output_path, **kwargs):
        # Track calls per output path
        self.call_count[output_path] = self.call_count.get(output_path, 0) + 1

        # Permanent failures always fail
        idx = int(os.path.basename(output_path).split(".")[0])
        if idx in self.permanent_fail_indices:
            return False

        # Transient failures: fail first time, succeed on retry
        if idx in self.fail_indices and self.call_count[output_path] == 1:
            return False

        # Write a minimal WAV header (enough for soundfile to read)
        _write_fake_wav(output_path, len(text) * 1000)
        return True


class CrashingTTSEngine:
    """Mock TTS engine that raises on specific indices."""

    def __init__(self, crash_indices=None):
        self.crash_indices = set(crash_indices or [])

    async def generate_line(self, text, output_path, **kwargs):
        idx = int(os.path.basename(output_path).split(".")[0])
        if idx in self.crash_indices:
            raise RuntimeError(f"OOM crash on index {idx}")
        _write_fake_wav(output_path, len(text) * 1000)
        return True


def _write_fake_wav(path, num_samples=24000):
    """Write a minimal valid WAV file."""
    import struct
    sample_rate = 24000
    num_channels = 1
    bits_per_sample = 16
    data_size = num_samples * num_channels * (bits_per_sample // 8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * num_channels * bits_per_sample // 8))
        f.write(struct.pack("<H", num_channels * bits_per_sample // 8))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


def _make_jobs(texts, speaker="TestChar"):
    """Create job tuples from a list of texts."""
    return [
        (i, {"text": t, "speaker": speaker})
        for i, t in enumerate(texts)
    ]


def _make_characters(speaker="TestChar", voice="test_voice"):
    return [{"name": speaker, "voice": voice, "voice_archetype": "男-普通"}]


def _speaker_map(speaker="TestChar", voice="test_voice"):
    return {speaker: voice}


class TestRatioBoundaries:
    """Test that ratio classification boundaries are correct."""

    def test_ratio_at_lower_bound(self):
        assert MIN_RATIO <= 0.8 <= MAX_RATIO

    def test_ratio_below_lower_bound(self):
        assert not (MIN_RATIO <= 0.79 <= MAX_RATIO)

    def test_ratio_at_upper_bound(self):
        assert MIN_RATIO <= 1.2 <= MAX_RATIO

    def test_ratio_above_upper_bound(self):
        assert not (MIN_RATIO <= 1.21 <= MAX_RATIO)

    def test_ratio_exactly_one(self):
        assert MIN_RATIO <= 1.0 <= MAX_RATIO


@pytest.mark.asyncio
class TestPipelineTermination:
    """Test that the pipeline terminates correctly."""

    async def test_all_pass(self):
        """5 jobs, all pass. Pipeline should complete with counter=0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MockTTSEngine()
            pipeline = QAPipeline(
                tts_engine=engine,
                audio_dir=tmpdir,
                num_workers=2,
            )
            # Disable whisper — just test TTS + termination
            pipeline._whisper_available = False

            jobs = _make_jobs(["你好世界" * 5] * 5)
            report = await pipeline.run(jobs, _speaker_map(), _make_characters())

            assert report["summary"]["total"] == 5
            assert pipeline.in_flight == 0

    async def test_empty_queue(self):
        """0 jobs after filtering. Pipeline completes immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MockTTSEngine()
            pipeline = QAPipeline(
                tts_engine=engine, audio_dir=tmpdir, num_workers=2,
            )
            pipeline._whisper_available = False

            jobs = []
            report = await pipeline.run(jobs, _speaker_map(), _make_characters())
            assert report["summary"]["total"] == 0


@pytest.mark.asyncio
class TestRetryFlow:
    """Test retry and escalation logic."""

    async def test_transient_failure_retried(self):
        """Job fails once, succeeds on retry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MockTTSEngine(fail_indices={0})
            pipeline = QAPipeline(
                tts_engine=engine, audio_dir=tmpdir, num_workers=1,
            )
            pipeline._whisper_available = False

            jobs = _make_jobs(["测试重试逻辑的文本"])
            report = await pipeline.run(jobs, _speaker_map(), _make_characters())

            # Should have retried and eventually completed
            assert report["summary"]["total"] == 1
            assert pipeline.in_flight == 0

    async def test_permanent_failure_escalated(self):
        """Job fails all retries, becomes human_review."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MockTTSEngine(permanent_fail_indices={0})
            pipeline = QAPipeline(
                tts_engine=engine, audio_dir=tmpdir, num_workers=1,
            )
            pipeline._whisper_available = False

            jobs = _make_jobs(["这个文本永远失败"])
            report = await pipeline.run(jobs, _speaker_map(), _make_characters())

            assert report["summary"]["total"] == 1
            assert report["summary"]["human_review"] == 1
            assert pipeline.in_flight == 0


@pytest.mark.asyncio
class TestEmptyLines:
    """Test that punctuation-only lines are auto-passed."""

    async def test_punctuation_only_skips_tts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MockTTSEngine()
            pipeline = QAPipeline(
                tts_engine=engine, audio_dir=tmpdir, num_workers=1,
            )
            pipeline._whisper_available = False

            jobs = _make_jobs(["……", "——。", "正常的文本内容"])
            report = await pipeline.run(jobs, _speaker_map(), _make_characters())

            # Punctuation lines should auto-pass without TTS
            pass_lines = [l for l in report["lines"] if l["status"] == "pass"]
            assert len(pass_lines) >= 2  # at least the 2 punctuation lines


@pytest.mark.asyncio
class TestWorkerCrashRecovery:
    """Test that a worker crash doesn't hang the pipeline."""

    async def test_crash_becomes_human_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CrashingTTSEngine(crash_indices={1})
            pipeline = QAPipeline(
                tts_engine=engine, audio_dir=tmpdir, num_workers=2,
            )
            pipeline._whisper_available = False

            jobs = _make_jobs(["正常文本", "会崩溃的文本", "另一个正常文本"])
            report = await pipeline.run(jobs, _speaker_map(), _make_characters())

            # All 3 should complete (no hang)
            assert report["summary"]["total"] == 3
            assert pipeline.in_flight == 0
            # The crashed one should be human_review
            review_lines = [l for l in report["lines"] if l["status"] == "human_review"]
            assert len(review_lines) >= 1


class TestPatternDB:
    """Test pattern DB aggregation."""

    def test_aggregate_creates_db(self):
        from gnosis.pattern_db import aggregate

        with tempfile.TemporaryDirectory() as tmpdir:
            qa_path = os.path.join(tmpdir, "qa.json")
            db_path = os.path.join(tmpdir, "pattern_db.json")

            qa_data = {
                "summary": {"total": 3, "pass": 2, "auto_retried": 0, "human_review": 1, "pass_rate": 0.667},
                "lines": [
                    {"index": 0, "text": "短文本", "status": "pass", "voice_seed": "logos", "script_char_len": 3,
                     "voice_tag": "男-普通"},
                    {"index": 1, "text": "这是一个比较长的测试文本用于测试中等长度的分桶逻辑",
                     "status": "human_review", "voice_seed": "logos", "script_char_len": 25,
                     "voice_tag": "男-普通"},
                    {"index": 2, "text": "普通文本", "status": "pass", "voice_seed": "duanya", "script_char_len": 4,
                     "voice_tag": "男-温柔"},
                ],
            }
            with open(qa_path, "w") as f:
                json.dump(qa_data, f)

            db = aggregate(qa_path, db_path)

            assert db["total_lines_processed"] == 3
            assert len(db["patterns"]) > 0
            assert os.path.exists(db_path)


class TestQAReport:
    """Test qa-report formatted output."""

    def test_format_report(self):
        from gnosis.qa import format_qa_report

        qa_data = {
            "summary": {"total": 100, "pass": 90, "auto_retried": 5, "human_review": 5, "pass_rate": 0.95},
            "lines": [
                {"index": i, "speaker": "Test", "text": f"Line {i}", "status": "pass",
                 "asr_char_len": 5, "script_char_len": 5, "ratio": 1.0, "retries": 0,
                 "retry_reasons": [], "audio_path": f"0{i:03d}.wav",
                 "voice_seed": "logos", "voice_tag": "男-普通", "punctuation_stripped": False}
                for i in range(95)
            ] + [
                {"index": 95 + i, "speaker": "Test", "text": f"Bad line {i}",
                 "status": "human_review", "asr_char_len": 2, "script_char_len": 10,
                 "ratio": 0.2, "retries": 2, "retry_reasons": ["ratio_0.200"],
                 "audio_path": f"0{95+i:03d}.wav", "voice_seed": "logos",
                 "voice_tag": "男-普通", "punctuation_stripped": False}
                for i in range(5)
            ],
        }

        report = format_qa_report(qa_data)
        assert "QA REPORT" in report
        assert "95.0%" in report
        assert "Human review" in report
