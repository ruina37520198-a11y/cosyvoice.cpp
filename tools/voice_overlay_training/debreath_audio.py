"""3-Band Spectral De-Breather for Close-Mic TTS Training Datasets.

Detects and smoothly attenuates isolated inter-clause inhale gasps in mono PCM16 WAV
recordings without distorting voiced speech or high-frequency sibilants (e.g., s/sh/ch/ts):
  - Band 1 (Low / Fundamental):   80–550 Hz   (must stay weak during unvoiced inhale)
  - Band 2 (Mid / Breath Hiss): 1200–4500 Hz  (dominant during sharp close-mic inhale)
  - Band 3 (High / Sibilants):  5500–11000 Hz (stronger on true s/sh/ts consonants)

Only spans that are:
  1. Unvoiced mid-band air hiss (high mid/low ratio, bounded high/mid sibilant ratio),
  2. Within typical inhale duration (45–480 ms), and
  3. Adjacent to a low-energy pause/silence boundary (so mid-word consonants stay untouched)
receive a smooth cosine-tapered gain reduction. Zero external dependencies beyond NumPy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import wave
from pathlib import Path

import numpy as np


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sw * 8}-bit: {path}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


def _write_wav_mono(path: Path, pcm: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(pcm, -1.0, 32767.0 / 32768.0)
    int16 = np.round(clipped * 32767.0).astype("<i2")
    tmp = path.with_name(f".{path.name}.tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    os.replace(tmp, path)


def detect_and_attenuate_breaths(
    pcm: np.ndarray,
    sr: int,
    *,
    attenuation_db: float = -22.0,
    min_breath_ms: float = 45.0,
    max_breath_ms: float = 480.0,
    fade_ms: float = 18.0,
    silence_db: float = -42.0,
    min_mid_db: float = -46.0,
    max_mid_db: float = -12.0,
    min_mid_over_low_db: float = 7.0,
    max_high_over_mid_db: float = 2.0,
    silence_context_ms: float = 120.0,
) -> tuple[np.ndarray, list[dict]]:
    """Return `(processed_pcm, detected_spans)` without modifying voiced speech or sibilants."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    n_samples = len(x)
    win_len = max(256, 1 << int(math.ceil(math.log2(max(1, int(sr * 0.025))))))
    hop_len = max(64, int(round(sr * 0.010)))
    if n_samples < win_len:
        return x.copy(), []

    n_frames = 1 + (n_samples - win_len) // hop_len
    idx = np.arange(win_len)[None, :] + (np.arange(n_frames) * hop_len)[:, None]
    window = np.hanning(win_len).astype(np.float32)
    frames = x[idx] * window[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win_len, d=1.0 / float(sr))

    def _band_db(f_lo: float, f_hi: float) -> np.ndarray:
        mask = (freqs >= f_lo) & (freqs <= min(f_hi, sr * 0.49))
        if not np.any(mask):
            return np.full(n_frames, -120.0, dtype=np.float32)
        power = np.mean(spec[:, mask], axis=1) / float(win_len)
        return (10.0 * np.log10(np.maximum(power, 1e-12))).astype(np.float32)

    low_db = _band_db(80.0, 550.0)
    mid_db = _band_db(1200.0, 4500.0)
    high_db = _band_db(5500.0, 11000.0)
    frame_rms_db = (20.0 * np.log10(np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12))).astype(np.float32)

    is_silence = frame_rms_db < float(silence_db)
    is_breath_frame = (
        (mid_db >= float(min_mid_db))
        & (mid_db <= float(max_mid_db))
        & ((mid_db - low_db) >= float(min_mid_over_low_db))
        & ((high_db - mid_db) <= float(max_high_over_mid_db))
        & (frame_rms_db < -14.0)
    )

    # Bridge single-frame dips (10ms) inside a continuous breath inhale
    bridged = is_breath_frame.copy()
    for i in range(1, n_frames - 1):
        if bridged[i - 1] and bridged[i + 1]:
            bridged[i] = True

    ctx_frames = max(1, int(round(silence_context_ms / 10.0)))
    min_f = max(2, int(round(min_breath_ms / 10.0)))
    max_f = max(min_f, int(round(max_breath_ms / 10.0)))

    spans: list[dict] = []
    i = 0
    while i < n_frames:
        if not bridged[i]:
            i += 1
            continue
        j = i
        while j < n_frames and bridged[j]:
            j += 1
        span_len = j - i
        if min_f <= span_len <= max_f:
            left_lo = max(0, i - ctx_frames)
            right_hi = min(n_frames, j + ctx_frames)
            near_edge = (i <= ctx_frames) or (j >= n_frames - ctx_frames)
            near_silence = near_edge or bool(np.any(is_silence[left_lo:i])) or bool(np.any(is_silence[j:right_hi]))
            if near_silence:
                start_s = i * hop_len
                end_s = min(n_samples, j * hop_len + win_len // 2)
                spans.append({
                    "start_sec": round(start_s / float(sr), 3),
                    "end_sec": round(end_s / float(sr), 3),
                    "duration_ms": round((end_s - start_s) * 1000.0 / float(sr), 1),
                    "start_sample": start_s,
                    "end_sample": end_s,
                })
        i = j

    if not spans:
        return x.copy(), []

    gain = np.ones(n_samples, dtype=np.float32)
    target_gain = float(10.0 ** (attenuation_db / 20.0))
    fade_n = max(1, int(round(sr * fade_ms / 1000.0)))

    for sp in spans:
        s0 = sp["start_sample"]
        s1 = sp["end_sample"]
        seg_len = s1 - s0
        if seg_len <= 0:
            continue
        f = min(fade_n, seg_len // 2)
        env = np.full(seg_len, target_gain, dtype=np.float32)
        if f > 1:
            t = np.linspace(0.0, math.pi, f, dtype=np.float32)
            # Cosine transition 1.0 -> target_gain and target_gain -> 1.0
            env[:f] = target_gain + (1.0 - target_gain) * 0.5 * (1.0 + np.cos(t))
            env[-f:] = target_gain + (1.0 - target_gain) * 0.5 * (1.0 - np.cos(t))
        gain[s0:s1] = np.minimum(gain[s0:s1], env)

    out = (x * gain).astype(np.float32)
    clean_spans = [{k: v for k, v in sp.items() if not k.endswith("_sample")} for sp in spans]
    return out, clean_spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True, help="Directory of input PCM16 WAVs (+ paired .txt)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (required unless --dry-run)")
    parser.add_argument("--attenuation-db", type=float, default=-22.0, help="Gain reduction in dB on detected inhales (default: -22)")
    parser.add_argument("--copy-sidecars", action="store_true", help="Copy matching .txt / .inst.txt / manifests to --out-dir")
    parser.add_argument("--dry-run", action="store_true", help="Report detected breath spans without writing WAV files")
    args = parser.parse_args()

    in_dir = args.in_dir.resolve()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")
    if not args.dry_run and args.out_dir is None:
        parser.error("--out-dir is required unless --dry-run is specified")
    out_dir = args.out_dir.resolve() if args.out_dir is not None else None
    if out_dir is not None and out_dir == in_dir:
        raise ValueError("--out-dir must differ from --in-dir to preserve original recordings")

    wavs = sorted(p for p in in_dir.rglob("*.wav") if p.is_file())
    if not wavs:
        raise SystemExit(f"No .wav files found under {in_dir}")

    report = []
    total_spans = 0
    for wav_path in wavs:
        rel = wav_path.relative_to(in_dir)
        pcm, sr = _read_wav_mono(wav_path)
        proc, spans = detect_and_attenuate_breaths(pcm, sr, attenuation_db=args.attenuation_db)
        total_spans += len(spans)
        report.append({"file": rel.as_posix(), "breath_count": len(spans), "spans": spans})
        if not args.dry_run and out_dir is not None:
            dst = out_dir / rel
            _write_wav_mono(dst, proc, sr)
            if args.copy_sidecars:
                for suffix in (".txt", ".inst.txt"):
                    src_sc = wav_path.with_name(wav_path.stem + suffix)
                    if src_sc.is_file():
                        dst_sc = dst.with_name(dst.stem + suffix)
                        dst_sc.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_sc, dst_sc)

    if not args.dry_run and out_dir is not None and args.copy_sidecars:
        for manifest_name in ("metadata.jsonl", "transcripts.txt", "list.txt", "metadata.txt", "metadata.csv"):
            m_src = in_dir / manifest_name
            if m_src.is_file():
                shutil.copy2(m_src, out_dir / manifest_name)
        (out_dir / "debreath_report.json").write_text(
            json.dumps({"total_files": len(wavs), "total_breaths": total_spans, "files": report}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({
        "dry_run": args.dry_run,
        "total_files": len(wavs),
        "files_with_breaths": sum(1 for r in report if r["breath_count"] > 0),
        "total_breaths": total_spans,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
