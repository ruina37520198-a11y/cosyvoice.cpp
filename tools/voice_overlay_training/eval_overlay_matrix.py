"""Stage 4: Free-Generation Evaluation Matrix for CosyVoice3 Voice Overlays.

Synthesizes a fixed multilingual + instruction + long-EOS prompt matrix across
multiple `(epoch_overlay.gguf, voice_adapter_scale, seed)` candidates plus the unmodified
Base baseline, measures duration/RTF/peak/clipping/tail-silence/EOS status, and generates
a Markdown listening report for human blind/side-by-side auditioning.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from gguf_utils import find_cosyvoice_cli


def resolve_cfg_path(raw: str | Path, cfg_dir: Path) -> Path:
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (cfg_dir / p).resolve()


DEFAULT_EVAL_CASES = [
    {
        "id": "jp_short_daily",
        "category": "JP-ZeroShot",
        "mode": "zero-shot",
        "instruction": "",
        "display_text": "おはようございます。今日も一日、よろしくお願いしますね。",
        "text": "オハヨウゴザイマス。キョウモイチニチ、ヨロシクオネガイシマスネ。",
    },
    {
        "id": "jp_emotion_happy",
        "category": "JP-Instruct",
        "mode": "instruct",
        "instruction": "用开心轻快的语气说。",
        "display_text": "わあ、本当ですか？それなら今すぐ一緒に行きましょう！",
        "text": "ワア、ホントウデスカ？ソレナライマスグイッショニイキマショウ！",
    },
    {
        "id": "jp_long_eos",
        "category": "JP-LongEOS",
        "mode": "zero-shot",
        "instruction": "",
        "display_text": "静かな朝に窓を開けると、やわらかい風が部屋に入ってきて、なんだか少しだけ気持ちまで軽くなる気がします。",
        "text": "シズカナアサニマドオアケルト、ヤワラカイカゼガヘヤニハイッテキテ、ナンダカスコシダケキモチマデカルクナルキガシマス。",
    },
    {
        "id": "zh_daily",
        "category": "ZH-CrossLingual",
        "mode": "cross-lingual",
        "instruction": "",
        "text": "今天天气真好，忙完这段工作之后我们一起去喝杯茶吧。",
    },
    {
        "id": "zh_inst_soft",
        "category": "ZH-Instruct",
        "mode": "instruct",
        "instruction": "用温柔轻声的语气说。",
        "text": "辛苦啦，累了的话就先休息一下，剩下的事不用着急。",
    },
    {
        "id": "zh_inst_happy",
        "category": "ZH-Instruct",
        "mode": "instruct",
        "instruction": "用开心轻快的语气说。",
        "text": "太棒了！这次调整之后听起来自然多了，你觉得呢？",
    },
    {
        "id": "zh_long_eos",
        "category": "ZH-LongEOS",
        "mode": "cross-lingual",
        "instruction": "",
        "text": "如果发现哪里听起来不太顺，就把那一句单独记下来，对比不同强度和不同轮次的效果，这样很快就能挑出最自然的版本。",
    },
    {
        "id": "mix_en_zh",
        "category": "Mixed-EN-ZH",
        "mode": "cross-lingual",
        "instruction": "",
        "text": "这个 LoRA Overlay 只需要三十多 MB，就能直接套在共享底座上使用。",
    },
]


def inspect_wav(wav_path: Path) -> dict:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sw != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sw * 8}-bit: {wav_path}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    dur = len(pcm) / float(sr) if sr > 0 else 0.0
    peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
    rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12)) if len(pcm) else 0.0
    clipped = int(np.count_nonzero(np.abs(pcm) >= 0.999))
    frame_len = max(1, int(round(sr * 0.02)))
    tail_silence_ms = 0.0
    if len(pcm) >= frame_len:
        n_frames = len(pcm) // frame_len
        frames = pcm[: n_frames * frame_len].reshape(n_frames, frame_len)
        fdb = 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1) + 1e-12))
        active = np.flatnonzero(fdb >= -45.0)
        if len(active) > 0:
            tail_silence_ms = round((n_frames - 1 - int(active[-1])) * 20.0, 1)
    return {
        "sample_rate": sr,
        "frames": len(pcm),
        "duration_sec": round(dur, 3),
        "rms_db": round(20.0 * math.log10(max(rms, 1e-6)), 2),
        "peak_db": round(20.0 * math.log10(max(peak, 1e-6)), 2),
        "clipped_samples": clipped,
        "tail_silence_ms": tail_silence_ms,
    }


def run_single_synthesis(
    cli_exe: Path,
    base_gguf: Path,
    prompt_speech: Path,
    out_wav: Path,
    *,
    text: str,
    instruction: str,
    mode: str | None = None,
    seed: int,
    backend: str | None,
    kv_cache_type: str | None,
    overlay_path: Path | None = None,
    scale: float = 1.0,
) -> dict:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli_exe),
        "--model", str(base_gguf),
        "--prompt-speech", str(prompt_speech),
        "--text", text,
        "--output", str(out_wav),
        "--seed", str(seed),
        "--strict-seed", "1",
    ]
    if backend:
        cmd.extend(["--backend", backend])
    if kv_cache_type:
        cmd.extend(["--llm-kv-cache-type", kv_cache_type])
    if mode:
        cmd.extend(["--mode", mode])
    if instruction:
        cmd.extend(["--instruction", instruction])
    if overlay_path is not None:
        cmd.extend(["--voice-adapter", str(overlay_path), "--voice-adapter-scale", f"{scale:g}"])

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cli_exe.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    wall_s = round(time.perf_counter() - t0, 3)
    combined_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    eos_limit_hit = "token limit reached before stop token" in combined_log.lower()

    if proc.returncode != 0 or not out_wav.is_file() or out_wav.stat().st_size < 1024:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "wall_sec": wall_s,
            "eos_limit_hit": eos_limit_hit,
            "error": combined_log.strip()[-350:],
            "wav": str(out_wav),
        }

    wav_info = inspect_wav(out_wav)
    rtf = round(wall_s / wav_info["duration_sec"], 3) if wav_info["duration_sec"] > 0 else None
    return {
        "ok": True,
        "returncode": 0,
        "wall_sec": wall_s,
        "rtf": rtf,
        "eos_limit_hit": eos_limit_hit,
        "wav": str(out_wav),
        **wav_info,
    }


def render_markdown_report(report: dict, md_path: Path) -> None:
    lines = [
        f"# CosyVoice3 Voice Overlay Evaluation Matrix: {report['voice_name']}",
        "",
        f"- **Base GGUF**: `{report['base_gguf']}`",
        f"- **Reference Prompt**: `{report['prompt_speech']}`",
        f"- **Backend / KV Cache / Seeds**: `{report['backend'] or 'auto'}` / `{report['kv_cache_type'] or 'default'}` / `seeds={report['seeds']}` (`strict_seed=1`)",
        "",
        "## Candidate Summary",
        "",
        "| Label | Overlay | Scale | Seed | OK/Total | EOS Limit | Clipped | Avg Dur (s) | Avg RTF |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cand in report["candidates"]:
        rows = cand["cases"]
        ok_rows = [r for r in rows if r["ok"]]
        eos_hits = sum(1 for r in rows if r.get("eos_limit_hit"))
        clips = sum(int(r.get("clipped_samples", 0)) for r in ok_rows)
        avg_dur = round(sum(r["duration_sec"] for r in ok_rows) / len(ok_rows), 2) if ok_rows else 0.0
        avg_rtf = round(sum(r["rtf"] for r in ok_rows if r.get("rtf")) / len(ok_rows), 3) if ok_rows else 0.0
        ov_name = Path(cand["overlay"]).name if cand["overlay"] else "(Base)"
        lines.append(
            f"| `{cand['label']}` | `{ov_name}` | `{cand['scale']}` | `{cand['seed']}` | "
            f"{len(ok_rows)}/{len(rows)} | {eos_hits} | {clips} | {avg_dur} | {avg_rtf} |"
        )

    lines.extend(["", "## Per-Case Audition Table", ""])
    for case in report["eval_cases"]:
        lines.append(f"### `{case['id']}` ({case['category']})")
        if case.get("mode"):
            lines.append(f"- Mode: `{case['mode']}`")
        if case.get("instruction"):
            lines.append(f"- Instruction: `{case['instruction']}`")
        if case.get("display_text"):
            lines.append(f"- Display Text: `{case['display_text']}`")
        lines.append(f"- Input Text: `{case['text']}`")
        lines.append("")
        lines.append("| Candidate | Status | Dur (s) | Tail Silence (ms) | Peak (dB) | WAV |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
        for cand in report["candidates"]:
            row = next(r for r in cand["cases"] if r["case_id"] == case["id"])
            status = "PASS" if (row["ok"] and not row.get("eos_limit_hit")) else ("EOS_LIMIT" if row.get("eos_limit_hit") else "FAIL")
            dur = row.get("duration_sec", "-")
            tail = row.get("tail_silence_ms", "-")
            peak = row.get("peak_db", "-")
            wav_rel = Path(row["wav"]).relative_to(md_path.parent).as_posix() if row["ok"] else ""
            wav_link = f"[{Path(row['wav']).name}]({wav_rel})" if row["ok"] else f"`{row.get('error', 'failed')[:60]}`"
            lines.append(f"| `{cand['label']}` | {status} | {dur} | {tail} | {peak} | {wav_link} |")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Stage 4 Free-Generation Evaluation Matrix for CosyVoice3 Voice Overlays")
    parser.add_argument("--config", type=Path, default=None, help="Optional train_config.json to auto-discover paths & overlays")
    parser.add_argument("--base", type=Path, default=None, help="Shared F16 or mixed HiFT-F32 base GGUF for inference")
    parser.add_argument("--runtime-dir", type=Path, default=None, help="Directory containing cosyvoice-cli")
    parser.add_argument("--prompt-speech", type=Path, default=None, help="Reference prompt_speech.gguf")
    parser.add_argument("--overlays", type=Path, nargs="*", default=None, help="Explicit list of overlay GGUF files")
    parser.add_argument("--scales", type=float, nargs="*", default=None, help="Scales to sweep per overlay (default: 0.55 0.7 0.85)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="One or more random seeds (e.g. --seeds 42 101 202)")
    parser.add_argument("--cases", type=Path, default=None, help="Optional JSON file defining custom eval cases")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for eval WAVs and reports")
    parser.add_argument("--backend", default=None, help="Optional cosyvoice-cli backend (e.g. vulkan0, cuda0, cpu)")
    parser.add_argument("--kv-cache-type", default=None, help="LLM KV cache type (default: f16)")
    parser.add_argument("--skip-base", action="store_true", help="Do not include unmodified Base baseline")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved matrix plan without running synthesis")
    args = parser.parse_args()

    cfg = {}
    cfg_dir = Path.cwd()
    if args.config is not None:
        cfg_path = args.config.resolve()
        cfg_dir = cfg_path.parent
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    paths = cfg.get("paths", {})
    eval_cfg = cfg.get("eval", {})
    voice_name = cfg.get("voice_name", "overlay_eval")

    if args.base is not None:
        base_gguf = args.base.resolve()
    elif paths.get("infer_base_gguf"):
        base_gguf = resolve_cfg_path(paths["infer_base_gguf"], cfg_dir)
    elif paths.get("base_gguf"):
        base_gguf = resolve_cfg_path(paths["base_gguf"], cfg_dir)
    else:
        parser.error("Missing --base (or paths.infer_base_gguf / paths.base_gguf in --config)")

    rt_dir = args.runtime_dir.resolve() if args.runtime_dir is not None else (
        resolve_cfg_path(paths["cosyvoice_runtime_dir"], cfg_dir) if paths.get("cosyvoice_runtime_dir") else None
    )

    if args.prompt_speech is not None:
        prompt_speech = args.prompt_speech.resolve()
    elif paths.get("processed_dataset_dir"):
        prompt_speech = resolve_cfg_path(paths["processed_dataset_dir"], cfg_dir) / "recommended_reference.gguf"
    else:
        parser.error("Missing --prompt-speech (or paths.processed_dataset_dir in --config)")

    if args.overlays is not None:
        overlays = [p.resolve() for p in args.overlays]
    elif paths.get("runs_dir"):
        run_dir = resolve_cfg_path(paths["runs_dir"], cfg_dir) / voice_name
        ckpt_dir = run_dir / "checkpoints"
        overlays = sorted(ckpt_dir.glob("epoch_*.gguf")) if ckpt_dir.is_dir() else []
        best_ov = run_dir / "best_overlay.gguf"
        if best_ov.is_file() and best_ov not in overlays:
            overlays.append(best_ov)
    else:
        overlays = []

    scales = args.scales if args.scales is not None else [float(s) for s in eval_cfg.get("scales", [0.55, 0.7, 0.85])]
    seeds = args.seeds if args.seeds is not None else [int(eval_cfg.get("seed", 42))]
    backend = args.backend if args.backend is not None else eval_cfg.get("backend")
    kv_cache_type = args.kv_cache_type if args.kv_cache_type is not None else eval_cfg.get("llm_kv_cache_type", "f16")

    eval_cases = (
        json.loads(args.cases.resolve().read_text(encoding="utf-8-sig"))
        if args.cases is not None
        else DEFAULT_EVAL_CASES
    )

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    elif paths.get("runs_dir"):
        out_dir = resolve_cfg_path(paths["runs_dir"], cfg_dir) / voice_name / "eval_matrix"
    else:
        out_dir = Path.cwd() / "eval_matrix"

    multi_seed = len(seeds) > 1
    candidates = []
    for seed in seeds:
        seed_suffix = f"_seed{seed}" if multi_seed else ""
        if not args.skip_base:
            candidates.append({"label": f"base{seed_suffix}", "overlay": None, "scale": 0.0, "seed": seed})
        for ov in overlays:
            for sc in scales:
                sc_str = format(sc, "g").replace(".", "p")
                candidates.append({
                    "label": f"{ov.stem}_s{sc_str}{seed_suffix}",
                    "overlay": str(ov),
                    "scale": sc,
                    "seed": seed,
                })

    plan = {
        "voice_name": voice_name,
        "runtime_dir": str(rt_dir) if rt_dir else "PATH",
        "base_gguf": str(base_gguf),
        "prompt_speech": str(prompt_speech),
        "backend": backend,
        "kv_cache_type": kv_cache_type,
        "seeds": seeds,
        "out_dir": str(out_dir),
        "candidate_count": len(candidates),
        "case_count": len(eval_cases),
        "total_runs": len(candidates) * len(eval_cases),
        "candidates": candidates,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    cli_exe = find_cosyvoice_cli(rt_dir)
    if not base_gguf.is_file():
        raise FileNotFoundError(f"Missing base GGUF: {base_gguf}")
    if not prompt_speech.is_file():
        raise FileNotFoundError(f"Missing prompt speech GGUF: {prompt_speech}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_candidates = []
    total = len(candidates) * len(eval_cases)
    done = 0

    for cand in candidates:
        cand_rows = []
        ov_path = Path(cand["overlay"]) if cand["overlay"] else None
        for case in eval_cases:
            done += 1
            wav_out = out_dir / "wav" / cand["label"] / f"{case['id']}.wav"
            res = run_single_synthesis(
                cli_exe,
                base_gguf,
                prompt_speech,
                wav_out,
                text=case["text"],
                instruction=case.get("instruction", ""),
                mode=case.get("mode"),
                seed=int(cand["seed"]),
                backend=backend,
                kv_cache_type=kv_cache_type,
                overlay_path=ov_path,
                scale=float(cand["scale"]),
            )
            res["case_id"] = case["id"]
            res["category"] = case.get("category", "")
            cand_rows.append(res)
            tag = "OK" if (res["ok"] and not res.get("eos_limit_hit")) else "WARN/FAIL"
            print(f"[{done}/{total}] {cand['label']} | {case['id']} -> {tag} (dur={res.get('duration_sec','-')}s, wall={res['wall_sec']}s)")
        results_candidates.append({**cand, "cases": cand_rows})

    report = {
        "voice_name": voice_name,
        "base_gguf": str(base_gguf),
        "prompt_speech": str(prompt_speech),
        "backend": backend,
        "kv_cache_type": kv_cache_type,
        "seeds": seeds,
        "eval_cases": eval_cases,
        "candidates": results_candidates,
    }
    json_path = out_dir / "eval_results.json"
    md_path = out_dir / "listening_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    render_markdown_report(report, md_path)
    print(f"Done! Results JSON: {json_path}\nListening Report: {md_path}")


if __name__ == "__main__":
    main()
