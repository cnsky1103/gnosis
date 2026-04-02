"""
ASR Model Benchmark — validates Whisper model accuracy for QA ratio thresholds.

Picks N random audio files from an existing project, transcribes with both
Whisper tiny and small, compares char count accuracy against the script.

Usage:
  python benchmark_asr.py --project 玩乐关系1 --samples 20
"""

import argparse
import json
import os
import random
import time

from gnosis.utils import clean_text, is_punctuation_only_text


def run_benchmark(project_name: str, num_samples: int = 20, models=("tiny", "small")):
    project_root = os.path.join("data", "projects", project_name)
    script_path = os.path.join(project_root, "script.json")
    audio_dir = os.path.join(project_root, "output_audio")

    with open(script_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    script_list = payload["script"]

    # Collect valid indices (non-empty text, audio exists)
    valid = []
    for i, line in enumerate(script_list):
        text = line.get("text", "")
        if is_punctuation_only_text(text) or len(clean_text(text)) == 0:
            continue
        wav = os.path.join(audio_dir, f"{i:04d}.wav")
        if not os.path.exists(wav):
            continue
        valid.append(i)

    if len(valid) < num_samples:
        print(f"Only {len(valid)} valid samples available, using all.")
        samples = valid
    else:
        samples = sorted(random.sample(valid, num_samples))

    print(f"Benchmarking {len(samples)} samples from {project_name}")
    print(f"Models: {models}")
    print("=" * 70)

    from faster_whisper import WhisperModel

    for model_name in models:
        print(f"\n--- Model: {model_name} ---")
        t0 = time.time()
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s")

        ratios = []
        errors = []
        t_start = time.time()

        for idx in samples:
            text = script_list[idx]["text"]
            wav = os.path.join(audio_dir, f"{idx:04d}.wav")
            script_len = len(clean_text(text))

            try:
                segments, _ = model.transcribe(wav, language="zh")
                asr_text = "".join([s.text for s in segments])
                asr_len = len(clean_text(asr_text))
            except Exception as e:
                print(f"  [{idx:04d}] ERROR: {e}")
                errors.append(idx)
                continue

            ratio = asr_len / script_len if script_len > 0 else 0
            ratios.append(ratio)

            status = "OK" if 0.8 <= ratio <= 1.2 else "FAIL"
            if status == "FAIL":
                print(
                    f"  [{idx:04d}] {status} ratio={ratio:.3f} "
                    f"script={script_len} asr={asr_len} "
                    f"text={text[:30]}..."
                )

        elapsed = time.time() - t_start
        print(f"\nResults for {model_name}:")
        print(f"  Samples: {len(ratios)}, Errors: {len(errors)}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample)")

        if ratios:
            in_range = sum(1 for r in ratios if 0.8 <= r <= 1.2)
            print(f"  In range (0.8-1.2): {in_range}/{len(ratios)} ({in_range/len(ratios):.1%})")
            print(f"  Mean ratio: {sum(ratios)/len(ratios):.3f}")
            print(f"  Min ratio:  {min(ratios):.3f}")
            print(f"  Max ratio:  {max(ratios):.3f}")

            # Char count error distribution
            diffs = [abs(r - 1.0) for r in ratios]
            avg_diff = sum(diffs) / len(diffs)
            print(f"  Avg abs deviation from 1.0: {avg_diff:.3f} ({avg_diff*100:.1f}%)")

            # False positive rate (good audio flagged as bad)
            false_flags = sum(1 for r in ratios if not (0.8 <= r <= 1.2))
            print(f"  Would-be-flagged: {false_flags}/{len(ratios)} ({false_flags/len(ratios):.1%})")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("  If tiny's 'would-be-flagged' rate is < 15%, use tiny (faster).")
    print("  If > 15%, use small (more accurate) or adjust thresholds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR model benchmark for QA thresholds")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    args = parser.parse_args()
    run_benchmark(args.project, args.samples)
