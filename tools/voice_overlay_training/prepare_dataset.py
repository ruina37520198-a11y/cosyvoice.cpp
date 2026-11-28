"""Stage 1: Prepare CosyVoice3-2512 audio + transcript dataset for Voice Overlay LoRA training.

Features:
  - Discovers paired audio + `.txt` (+ optional `.inst.txt`) or manifest files
    (`metadata.jsonl`, `transcripts.txt`, `list.txt`, `metadata.txt`, `metadata.csv`)
  - Optional waveform preprocessing: 3-band spectral de-breathing, lead/tail silence
    trimming with cosine fades, and active RMS loudness normalization with peak ceiling
  - Extracts 25 Hz FSQ speech tokens via `cosyvoice-cli --frontend-only`
  - Encodes transcripts & instructions via the native `cosyvoice.cpp` BPE tokenizer
  - Builds deterministic leakage-free group train/validation splits (`split.json`)
  - Ranks top-5 clean reference prompt candidates (`reference_candidates.json`)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from debreath_audio import detect_and_attenuate_breaths
from gguf_utils import (
    NativeCosyVoiceTokenizer,
    find_cosyvoice_cli,
    find_ffmpeg_binary,
    load_gguf_numpy,
    normalize_transcript_text,
    parse_gguf_tensors,
)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
RESERVED_TXT_NAMES = {
    "transcripts.txt",
    "list.txt",
    "metadata.txt",
    "readme.txt",
    "format_guide.txt",
}
TARGET_SR = 24000


def resolve_cfg_path(raw: str | Path, cfg_dir: Path) -> Path:
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (cfg_dir / p).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_audio_duration(audio_path: Path, ffprobe_exe: Path | None) -> float | None:
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                sr = wf.getframerate()
                if sr > 0:
                    return wf.getnframes() / float(sr)
        except (wave.Error, OSError):
            pass
    if ffprobe_exe is None or not ffprobe_exe.is_file():
        return None
    cmd = [
        str(ffprobe_exe),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if res.returncode != 0:
        return None
    try:
        return float(res.stdout.strip())
    except ValueError:
        return None


def load_audio_mono_24k(audio_path: Path, ffmpeg_exe: Path | None) -> tuple[np.ndarray, int]:
    """Decode any supported audio into float32 mono 24 kHz PCM in [-1.0, 1.0]."""
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == TARGET_SR:
                    raw = wf.readframes(wf.getnframes())
                    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                    return pcm, TARGET_SR
        except (wave.Error, OSError):
            pass

    if ffmpeg_exe is None or not ffmpeg_exe.is_file():
        raise RuntimeError(
            f"ffmpeg is required to decode/resample {audio_path.name} to 24 kHz mono PCM16, but was not found."
        )
    with tempfile.TemporaryDirectory(prefix="cv_decode_") as td:
        tmp_wav = Path(td) / "decoded.wav"
        cmd = [
            str(ffmpeg_exe), "-y", "-v", "error",
            "-i", str(audio_path),
            "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16",
            str(tmp_wav),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode != 0 or not tmp_wav.is_file():
            raise RuntimeError(f"ffmpeg decode failed for {audio_path.name}: {(proc.stderr or '').strip()[-240:]}")
        with wave.open(str(tmp_wav), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            return pcm, TARGET_SR


def write_wav_pcm16(path: Path, pcm: np.ndarray, sr: int = TARGET_SR) -> None:
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


def normalize_audio_waveform(
    pcm: np.ndarray,
    sr: int = TARGET_SR,
    *,
    trim_silence: bool = True,
    silence_threshold_db: float = -42.0,
    relative_threshold_db: float = 35.0,
    lead_ms: float = 75.0,
    tail_ms: float = 150.0,
    fade_ms: float = 15.0,
    normalize_loudness: bool = True,
    target_rms_db: float = -20.0,
    peak_ceiling_db: float = -1.0,
    debreath_enabled: bool = False,
    debreath_attenuation_db: float = -22.0,
) -> tuple[np.ndarray, dict]:
    """Normalize lead/tail silence, optional close-mic inhales, and active loudness without clipping."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    orig_len = len(x)
    if orig_len == 0:
        return x, {
            "orig_duration_sec": 0.0,
            "trimmed_lead_ms": 0.0,
            "trimmed_tail_ms": 0.0,
            "active_rms_db": -120.0,
            "peak_db": -120.0,
            "clipped_samples": 0,
            "silence_ratio": 1.0,
            "breaths_attenuated": 0,
        }

    # Remove DC offset gently
    x = x - float(np.mean(x))

    breaths_attenuated = 0
    if debreath_enabled:
        x, spans = detect_and_attenuate_breaths(x, sr, attenuation_db=debreath_attenuation_db)
        breaths_attenuated = len(spans)

    frame_len = max(1, int(round(sr * 0.020)))
    hop_len = max(1, int(round(sr * 0.010)))
    if len(x) <= frame_len:
        frame_rms = np.array([float(np.sqrt(np.mean(x * x) + 1e-12))], dtype=np.float32)
    else:
        n_frames = 1 + (len(x) - frame_len) // hop_len
        idx = np.arange(frame_len)[None, :] + (np.arange(n_frames) * hop_len)[:, None]
        frames = x[idx]
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)

    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-6))
    peak_frame_db = float(np.max(frame_db))
    eff_thresh = min(float(silence_threshold_db), peak_frame_db - float(relative_threshold_db))
    active_mask = frame_db >= eff_thresh

    trimmed_lead_ms = 0.0
    trimmed_tail_ms = 0.0
    if trim_silence and np.any(active_mask):
        active_indices = np.flatnonzero(active_mask)
        first_active_sample = int(active_indices[0]) * hop_len
        last_active_sample = min(len(x), int(active_indices[-1]) * hop_len + frame_len)

        lead_samples = max(0, int(round(sr * lead_ms / 1000.0)))
        tail_samples = max(0, int(round(sr * tail_ms / 1000.0)))

        start = first_active_sample - lead_samples
        end = last_active_sample + tail_samples

        pad_left = max(0, -start)
        pad_right = max(0, end - len(x))
        slice_start = max(0, start)
        slice_end = min(len(x), end)

        trimmed_lead_ms = round((slice_start - pad_left) * 1000.0 / sr, 2)
        trimmed_tail_ms = round(((len(x) - slice_end) - pad_right) * 1000.0 / sr, 2)

        core = x[slice_start:slice_end].copy()
        fade_n = min(int(round(sr * fade_ms / 1000.0)), len(core) // 4)
        if fade_n > 1:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fade_n, dtype=np.float32)))
            core[:fade_n] *= ramp
            core[-fade_n:] *= ramp[::-1]
        if pad_left or pad_right:
            x = np.concatenate([
                np.zeros(pad_left, dtype=np.float32),
                core,
                np.zeros(pad_right, dtype=np.float32),
            ])
        else:
            x = core

    if len(x) <= frame_len:
        post_rms = np.array([float(np.sqrt(np.mean(x * x) + 1e-12))], dtype=np.float32)
    else:
        n_frames = 1 + (len(x) - frame_len) // hop_len
        idx = np.arange(frame_len)[None, :] + (np.arange(n_frames) * hop_len)[:, None]
        frames = x[idx]
        post_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)
    post_db = 20.0 * np.log10(np.maximum(post_rms, 1e-6))
    post_active = post_db >= eff_thresh
    active_rms = (
        float(np.sqrt(np.mean(post_rms[post_active] ** 2)))
        if np.any(post_active)
        else float(np.sqrt(np.mean(x * x) + 1e-12))
    )

    if normalize_loudness and active_rms > 1e-5:
        target_lin = 10.0 ** (target_rms_db / 20.0)
        ceiling_lin = 10.0 ** (peak_ceiling_db / 20.0)
        gain = target_lin / active_rms
        peak = float(np.max(np.abs(x)))
        if peak * gain > ceiling_lin and peak > 0.0:
            gain = ceiling_lin / peak
        gain = min(gain, 10.0 ** (12.0 / 20.0))
        x = x * float(gain)
        active_rms *= float(gain)

    final_peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clipped_samples = int(np.count_nonzero(np.abs(x) >= 0.999))
    silence_ratio = float(1.0 - np.mean(post_active)) if len(post_active) else 1.0

    metrics = {
        "orig_duration_sec": round(orig_len / float(sr), 3),
        "trimmed_lead_ms": trimmed_lead_ms,
        "trimmed_tail_ms": trimmed_tail_ms,
        "active_rms_db": round(20.0 * math.log10(max(active_rms, 1e-6)), 2),
        "peak_db": round(20.0 * math.log10(max(final_peak, 1e-6)), 2),
        "clipped_samples": clipped_samples,
        "silence_ratio": round(silence_ratio, 4),
        "breaths_attenuated": breaths_attenuated,
    }
    return x.astype(np.float32), metrics


def discover_raw_entries(raw_dir: Path) -> list[dict]:
    """Discover `(audio_path, transcript_text, optional_instruction)` from manifests and paired files."""
    if not raw_dir.exists():
        return []

    entries_by_audio: dict[Path, dict] = {}

    def _set_entry(ap: Path, text: str, instruction: str = ""):
        entries_by_audio[ap] = {"text": text.strip(), "instruction": instruction.strip()}

    jsonl_path = raw_dir / "metadata.jsonl"
    if jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            rel_audio = item.get("audio") or item.get("file") or item.get("wav") or item.get("path")
            text = item.get("text") or item.get("transcript") or ""
            inst = item.get("instruction") or item.get("instructions") or ""
            if rel_audio and text.strip():
                ap = (raw_dir / rel_audio).resolve()
                if ap.is_file():
                    _set_entry(ap, text, str(inst))

    for list_name in ("transcripts.txt", "list.txt", "metadata.txt"):
        list_path = raw_dir / list_name
        if not list_path.is_file():
            continue
        for line in list_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            inst = ""
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                rel_audio = parts[0]
                if len(parts) >= 5:
                    text, inst = parts[3], parts[4]
                elif len(parts) >= 4:
                    text = parts[3]
                else:
                    text = parts[1]
            elif "\t" in line:
                parts = [p.strip() for p in line.split("\t")]
                rel_audio = parts[0]
                text = parts[1] if len(parts) >= 2 else ""
                inst = parts[2] if len(parts) >= 3 else ""
            else:
                continue
            ap = (raw_dir / rel_audio).resolve()
            if not ap.is_file():
                ap = (raw_dir / Path(rel_audio).name).resolve()
            if ap.is_file() and text:
                _set_entry(ap, text, inst)

    csv_path = raw_dir / "metadata.csv"
    if csv_path.is_file():
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                if row[0].strip().lower() in ("audio", "file", "filename", "path", "wav"):
                    continue
                rel_audio = row[0].strip()
                text = row[1].strip() if len(row) == 2 else row[-1].strip()
                inst = ""
                if len(row) >= 3 and row[1].strip():
                    text, inst = row[1].strip(), row[2].strip()
                ap = (raw_dir / rel_audio).resolve()
                if ap.is_file() and text:
                    _set_entry(ap, text, inst)

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        resolved = path.resolve()
        if resolved in entries_by_audio:
            continue
        txt_path = path.with_suffix(".txt")
        if txt_path.is_file() and txt_path.name.lower() not in RESERVED_TXT_NAMES:
            text = txt_path.read_text(encoding="utf-8-sig").strip()
            inst_path = path.with_name(path.stem + ".inst.txt")
            inst = inst_path.read_text(encoding="utf-8-sig").strip() if inst_path.is_file() else ""
            if text:
                _set_entry(resolved, text, inst)

    results = []
    for audio_path, info in sorted(entries_by_audio.items(), key=lambda kv: str(kv[0])):
        rel = audio_path.relative_to(raw_dir.resolve())
        sample_id = rel.with_suffix("").as_posix().replace("/", "__")
        results.append({
            "id": sample_id,
            "audio_path": audio_path,
            "raw_text": info["text"],
            "instruction": info["instruction"],
        })
    return results


def extract_single_clip(
    entry: dict,
    cli_exe: Path,
    ffprobe_exe: Path | None,
    ffmpeg_exe: Path | None,
    speech_tokenizer_onnx: Path,
    campplus_onnx: Path,
    prompt_cache_dir: Path,
    norm_wav_dir: Path,
    min_dur: float,
    max_dur: float,
    simplify_chinese: bool,
    preprocess_cfg: dict | None,
) -> dict:
    audio_path: Path = entry["audio_path"]
    raw_text: str = entry["raw_text"]
    raw_inst: str = entry.get("instruction", "")
    norm_text = normalize_transcript_text(raw_text, simplify_chinese=simplify_chinese)
    norm_inst = normalize_transcript_text(raw_inst, simplify_chinese=simplify_chinese) if raw_inst else ""
    if not norm_text:
        return {"id": entry["id"], "status": "skipped", "reason": "empty_transcript"}

    audio_metrics = None
    effective_audio = audio_path

    if preprocess_cfg and preprocess_cfg.get("enabled", False):
        pcm, sr = load_audio_mono_24k(audio_path, ffmpeg_exe)
        norm_pcm, audio_metrics = normalize_audio_waveform(
            pcm,
            sr,
            trim_silence=bool(preprocess_cfg.get("trim_silence", True)),
            silence_threshold_db=float(preprocess_cfg.get("silence_threshold_db", -42.0)),
            relative_threshold_db=float(preprocess_cfg.get("relative_threshold_db", 35.0)),
            lead_ms=float(preprocess_cfg.get("lead_ms", 75.0)),
            tail_ms=float(preprocess_cfg.get("tail_ms", 150.0)),
            fade_ms=float(preprocess_cfg.get("fade_ms", 15.0)),
            normalize_loudness=bool(preprocess_cfg.get("normalize_loudness", True)),
            target_rms_db=float(preprocess_cfg.get("target_rms_db", -20.0)),
            peak_ceiling_db=float(preprocess_cfg.get("peak_ceiling_db", -1.0)),
            debreath_enabled=bool(preprocess_cfg.get("debreath_enabled", False)),
            debreath_attenuation_db=float(preprocess_cfg.get("debreath_attenuation_db", -22.0)),
        )
        dur = len(norm_pcm) / float(sr)
        if dur < min_dur:
            return {"id": entry["id"], "status": "skipped", "reason": f"too_short_after_trim ({dur:.2f}s < {min_dur}s)"}
        if dur > max_dur:
            return {"id": entry["id"], "status": "skipped", "reason": f"too_long_after_trim ({dur:.2f}s > {max_dur}s)"}
        effective_audio = norm_wav_dir / f"{entry['id']}.wav"
        write_wav_pcm16(effective_audio, norm_pcm, sr)
    else:
        dur = probe_audio_duration(audio_path, ffprobe_exe)
        if dur is not None:
            if dur < min_dur:
                return {"id": entry["id"], "status": "skipped", "reason": f"too_short ({dur:.2f}s < {min_dur}s)"}
            if dur > max_dur:
                return {"id": entry["id"], "status": "skipped", "reason": f"too_long ({dur:.2f}s > {max_dur}s)"}

    audio_sha = sha256_file(effective_audio)
    text_sha = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
    cache_key = hashlib.sha256(f"{audio_sha}:{text_sha}".encode("utf-8")).hexdigest()[:16]

    gguf_out = prompt_cache_dir / f"{entry['id']}_{cache_key}.gguf"
    if not gguf_out.is_file():
        tmp_out = prompt_cache_dir / f".tmp_{entry['id']}_{cache_key}.gguf"
        if tmp_out.exists():
            tmp_out.unlink()
        cmd = [
            str(cli_exe),
            "--frontend-only",
            "--speech-tokenizer", str(speech_tokenizer_onnx),
            "--campplus", str(campplus_onnx),
            "--prompt-audio", str(effective_audio),
            "--prompt-text", norm_text,
            "--prompt-speech-output", str(tmp_out),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(cli_exe.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode != 0 or not tmp_out.is_file():
            err_tail = (proc.stderr or proc.stdout or "").strip()[-300:]
            return {
                "id": entry["id"],
                "status": "failed",
                "reason": f"cosyvoice-cli exit {proc.returncode}: {err_tail}",
            }
        os.replace(tmp_out, gguf_out)

    tensors, data_offset = parse_gguf_tensors(gguf_out)
    raw_bytes = gguf_out.read_bytes()
    speech_tokens = load_gguf_numpy(raw_bytes, data_offset, tensors["tokens"]).astype("int64").tolist()
    speech_dur = len(speech_tokens) / 25.0
    if dur is None:
        dur = speech_dur
    if speech_dur < min_dur or speech_dur > max_dur:
        return {
            "id": entry["id"],
            "status": "skipped",
            "reason": f"speech_tokens_duration_out_of_range ({speech_dur:.2f}s)",
        }

    result = {
        "id": entry["id"],
        "status": "ok",
        "audio_path": str(effective_audio),
        "raw_audio_path": str(audio_path),
        "prompt_gguf_path": str(gguf_out),
        "raw_text": raw_text,
        "norm_text": norm_text,
        "duration_sec": round(float(dur), 3),
        "speech_tokens": speech_tokens,
    }
    if norm_inst:
        result["instruction"] = norm_inst
    if audio_metrics is not None:
        result["audio_metrics"] = audio_metrics
    return result


def score_reference_candidate(sample: dict) -> float:
    """Higher score = cleaner, better-balanced ~5.8s neutral reference candidate for DiT Flow."""
    dur = float(sample["duration_sec"])
    score = 100.0
    score -= abs(dur - 5.8) * 6.0
    if dur < 3.5 or dur > 9.0:
        score -= 40.0
    if sample.get("instruction"):
        score -= 10.0
    m = sample.get("audio_metrics")
    if m:
        score -= abs(float(m.get("active_rms_db", -20.0)) - (-20.0)) * 2.0
        if int(m.get("clipped_samples", 0)) > 0:
            score -= 50.0
        if float(m.get("peak_db", -3.0)) > -0.5:
            score -= 15.0
        score -= float(m.get("silence_ratio", 0.15)) * 30.0
    return round(score, 3)


def build_group_split(samples: list[dict], val_ratio: float, seed: int) -> dict:
    """Group `_partN` clips and identical normalized transcripts together to prevent train/val leakage."""
    parent = {s["id"]: s["id"] for s in samples}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    seen_source: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    for sample in samples:
        sid = sample["id"]
        source = re.sub(r"_part[0-9]+$", "", sid)
        text = sample["norm_text"]
        if source in seen_source:
            union(sid, seen_source[source])
        if text in seen_text:
            union(sid, seen_text[text])
        seen_source[source] = sid
        seen_text[text] = sid

    groups: dict[str, list[dict]] = {}
    for sample in samples:
        groups.setdefault(find(sample["id"]), []).append(sample)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    val_target = max(1, int(round(len(samples) * val_ratio))) if len(samples) >= 4 else 0
    val_ids: list[str] = []
    train_ids: list[str] = []
    for key in keys:
        bucket = val_ids if len(val_ids) < val_target else train_ids
        bucket.extend(s["id"] for s in groups[key])
    if not train_ids:
        train_ids = list(val_ids)
        val_ids = val_ids[:1]
    return {
        "seed": seed,
        "train": train_ids,
        "validation": val_ids,
        "group_by": "source_parts_and_identical_normalized_text",
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare CosyVoice3-2512 audio + transcript dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to train_config.json")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg_dir = cfg_path.parent
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    paths = cfg["paths"]
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg.get("training", {})

    raw_dir = resolve_cfg_path(paths["raw_dataset_dir"], cfg_dir)
    proc_dir = resolve_cfg_path(paths["processed_dataset_dir"], cfg_dir)
    prompt_cache_dir = proc_dir / "prompt_gguf"
    norm_wav_dir = proc_dir / "normalized_wav"
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)

    rt_dir = resolve_cfg_path(paths["cosyvoice_runtime_dir"], cfg_dir)
    cli_exe = find_cosyvoice_cli(rt_dir)
    ffprobe_exe = find_ffmpeg_binary("ffprobe", rt_dir)
    ffmpeg_exe = find_ffmpeg_binary("ffmpeg", rt_dir)
    base_gguf = resolve_cfg_path(paths["base_gguf"], cfg_dir)
    speech_tokenizer_onnx = resolve_cfg_path(paths["speech_tokenizer_onnx"], cfg_dir)
    campplus_onnx = resolve_cfg_path(paths["campplus_onnx"], cfg_dir)
    preprocess_cfg = ds_cfg.get("audio_preprocess")

    entries = discover_raw_entries(raw_dir)
    if not entries:
        print(f"[ERROR] No paired audio + transcript files found in {raw_dir}")
        sys.exit(1)

    pre_enabled = bool(preprocess_cfg and preprocess_cfg.get("enabled"))
    deb_enabled = bool(preprocess_cfg and preprocess_cfg.get("debreath_enabled"))
    print(f"[1/4] Found {len(entries)} clips | preprocess={'ON' if pre_enabled else 'OFF'} | debreath={'ON' if deb_enabled else 'OFF'}")

    workers = max(1, int(ds_cfg.get("parallel_workers", 4)))
    min_dur = float(ds_cfg.get("min_duration_sec", 1.2))
    max_dur = float(ds_cfg.get("max_duration_sec", 18.0))
    simplify_zh = bool(ds_cfg.get("normalize_chinese_to_simplified", True))

    valid_samples = []
    skipped_samples = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(
                extract_single_clip,
                entry,
                cli_exe,
                ffprobe_exe,
                ffmpeg_exe,
                speech_tokenizer_onnx,
                campplus_onnx,
                prompt_cache_dir,
                norm_wav_dir,
                min_dur,
                max_dur,
                simplify_zh,
                preprocess_cfg,
            )
            for entry in entries
        ]
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            if res["status"] == "ok":
                valid_samples.append(res)
                inst_tag = " [inst]" if res.get("instruction") else ""
                print(
                    f"  [{idx}/{len(entries)}] OK: {res['id']}{inst_tag} "
                    f"({res['duration_sec']:.2f}s, {len(res['speech_tokens'])} speech tokens)"
                )
            else:
                skipped_samples.append(res)
                print(f"  [{idx}/{len(entries)}] {res['status'].upper()}: {res['id']} -> {res['reason']}")

    valid_samples.sort(key=lambda x: x["id"])
    if not valid_samples:
        print("[ERROR] No samples passed validation and frontend extraction.")
        sys.exit(1)

    print("[2/4] Encoding text & instructions via native cosyvoice.cpp BPE tokenizer...")
    instruction_prefix = ds_cfg.get("instruction_prefix", "You are a helpful assistant.")
    with NativeCosyVoiceTokenizer(rt_dir, base_gguf) as tokenizer:
        prefix_tokens = tokenizer.encode(f"{instruction_prefix}<|endofprompt|>")
        for sample in valid_samples:
            sample["text_tokens"] = tokenizer.encode(sample["norm_text"])
            sample["full_text_tokens"] = tokenizer.encode(
                f"{instruction_prefix}<|endofprompt|>{sample['norm_text']}"
            )
            if sample.get("instruction"):
                sample["instruct_prefix_tokens"] = tokenizer.encode(
                    f"{instruction_prefix} {sample['instruction']}<|endofprompt|>"
                )

    print("[3/4] Ranking reference prompt candidates and creating group-isolated split.json...")
    ranked = sorted(valid_samples, key=score_reference_candidate, reverse=True)
    top_k = [
        {
            "rank": rank_idx,
            "id": s["id"],
            "score": score_reference_candidate(s),
            "duration_sec": s["duration_sec"],
            "audio_path": s["audio_path"],
            "prompt_gguf_path": s["prompt_gguf_path"],
            "raw_text": s["raw_text"],
            "audio_metrics": s.get("audio_metrics"),
        }
        for rank_idx, s in enumerate(ranked[:5], 1)
    ]
    (proc_dir / "reference_candidates.json").write_text(
        json.dumps(top_k, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best_ref = ranked[0]
    rec_gguf = proc_dir / "recommended_reference.gguf"
    rec_txt = proc_dir / "recommended_reference.txt"
    shutil.copy2(best_ref["prompt_gguf_path"], rec_gguf)
    rec_txt.write_text(best_ref["raw_text"], encoding="utf-8")

    seed = int(tr_cfg.get("seed", 42))
    val_ratio = float(ds_cfg.get("val_ratio", 0.1))
    split_path = proc_dir / "split.json"
    computed_split = build_group_split(valid_samples, val_ratio=val_ratio, seed=seed)
    if split_path.is_file():
        existing = json.loads(split_path.read_text(encoding="utf-8-sig"))
        existing_ids = set(existing.get("train", [])) | set(existing.get("validation", []))
        current_ids = {s["id"] for s in valid_samples}
        if existing_ids != current_ids:
            split_path.write_text(json.dumps(computed_split, ensure_ascii=False, indent=2), encoding="utf-8")
            active_split = computed_split
        else:
            active_split = existing
    else:
        split_path.write_text(json.dumps(computed_split, ensure_ascii=False, indent=2), encoding="utf-8")
        active_split = computed_split

    total_dur = sum(s["duration_sec"] for s in valid_samples)
    dataset_payload = {
        "schema": 2,
        "voice_name": cfg.get("voice_name", "custom_voice"),
        "instruction_prefix": instruction_prefix,
        "prefix_tokens": prefix_tokens,
        "sos_token_id": 6561,
        "eos_token_id": 6562,
        "task_token_id": 6563,
        "audio_preprocess": preprocess_cfg or {"enabled": False},
        "recommended_reference": {
            "id": best_ref["id"],
            "score": score_reference_candidate(best_ref),
            "audio_path": best_ref["audio_path"],
            "prompt_gguf_path": str(rec_gguf),
            "raw_text": best_ref["raw_text"],
            "norm_text": best_ref["norm_text"],
            "duration_sec": best_ref["duration_sec"],
        },
        "reference_candidates_file": str(proc_dir / "reference_candidates.json"),
        "split_file": str(split_path),
        "samples": valid_samples,
        "skipped": skipped_samples,
    }

    out_json = proc_dir / "dataset_manifest.json"
    out_json.write_text(json.dumps(dataset_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[4/4] Dataset preparation complete:\n"
        f"  - Valid clips: {len(valid_samples)} (skipped/failed: {len(skipped_samples)})\n"
        f"  - Split: train={len(active_split['train'])}, val={len(active_split['validation'])} ({split_path.name})\n"
        f"  - Total audio: {total_dur:.1f}s ({total_dur / 60.0:.2f} min)\n"
        f"  - Top reference (#1/{len(top_k)}): {best_ref['id']} ({best_ref['duration_sec']:.2f}s, score={score_reference_candidate(best_ref)})\n"
        f"  - Manifest: {out_json}"
    )


if __name__ == "__main__":
    main()
