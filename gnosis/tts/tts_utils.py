import os
import wave
from gnosis.utils import collect_sorted_segments

PUNCTUATION_ONLY_SILENCE_MS = 400
DEFAULT_AUDIO_SAMPLE_RATE = 24000
DEFAULT_SOVITS_URL = "http://127.0.0.1:9880"

def filter_script_jobs_by_character(script_lines, character_name):
    target_character = (character_name or "").strip()
    jobs = []
    available_speakers = set()
    for i, line in enumerate(script_lines):
        speaker = str(line.get("speaker", "")).strip()
        if speaker:
            available_speakers.add(speaker)
        if not target_character or speaker == target_character:
            jobs.append((i, line))
    return jobs, available_speakers

def write_silence_wav(output_path, duration_ms, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE):
    frame_count = max(1, int(round(sample_rate * (duration_ms / 1000.0))))
    silence_bytes = b"\x00\x00" * frame_count  # mono int16
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(silence_bytes)

def delete_character_audio_segments(audio_dir, script_lines, character_name):
    target_character = (character_name or "").strip()
    if not target_character:
        return {"target": "", "matched_lines": 0, "deleted_files": 0}

    target_indices = {
        i
        for i, line in enumerate(script_lines)
        if isinstance(line, dict) and str(line.get("speaker", "")).strip() == target_character
    }
    if not target_indices:
        return {
            "target": target_character,
            "matched_lines": 0,
            "deleted_files": 0,
        }
    if not os.path.isdir(audio_dir):
        return {
            "target": target_character,
            "matched_lines": len(target_indices),
            "deleted_files": 0,
        }

    deleted_files = 0
    for audio_path in collect_sorted_segments(audio_dir):
        file_name = os.path.basename(audio_path)
        stem = os.path.splitext(file_name)[0]
        if not stem.isdigit():
            continue
        if int(stem) not in target_indices:
            continue
        os.remove(audio_path)
        deleted_files += 1

    return {
        "target": target_character,
        "matched_lines": len(target_indices),
        "deleted_files": deleted_files,
    }



def is_han_char(ch):
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0x2CEB0 <= cp <= 0x2EBEF
    )
