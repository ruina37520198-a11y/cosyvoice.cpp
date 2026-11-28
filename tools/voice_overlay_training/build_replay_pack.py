"""Build a reusable Base-model Self-Distillation Replay Pack for CosyVoice3 LoRA training.

Prevents catastrophic forgetting of multilingual pronunciation, instruction following,
and EOS termination when fine-tuning a small single-speaker LoRA Voice Overlay.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gguf_utils import (
    NativeCosyVoiceTokenizer,
    find_cosyvoice_cli,
    find_ffmpeg_binary,
    load_gguf_numpy,
    normalize_transcript_text,
    parse_gguf_tensors,
)
from prepare_dataset import (
    discover_raw_entries,
    load_audio_mono_24k,
    normalize_audio_waveform,
    resolve_cfg_path,
    write_wav_pcm16,
)

DEFAULT_REPLAY_PROMPTS = [
    # Train replay: Chinese neutral & conversational
    {"id": "zh_daily_01", "split": "train", "seed": 101, "instruction": "", "text": "今天的天气真不错，我们下午一起出去散散步吧。"},
    {"id": "zh_daily_02", "split": "train", "seed": 102, "instruction": "", "text": "别急，先把刚刚的资料整理好，确认没有遗漏再发出去。"},
    {"id": "zh_daily_03", "split": "train", "seed": 103, "instruction": "", "text": "这个方案看起来很稳妥，不过实际测试的数据还是要再核对一次。"},
    {"id": "zh_daily_04", "split": "train", "seed": 104, "instruction": "", "text": "你最近睡得好吗？晚上记得早点休息，别又熬夜到天亮了。"},
    # Train replay: Chinese with explicit emotion/style instructions
    {"id": "zh_inst_happy", "split": "train", "seed": 105, "instruction": "用开心轻快的语气说。", "text": "太好了！这次测试竟然一次就全部通过了，真的超开心的！"},
    {"id": "zh_inst_soft", "split": "train", "seed": 106, "instruction": "用温柔轻声的语气说。", "text": "没关系的，累了就先靠着休息一下，剩下的事情交给我来就好。"},
    {"id": "zh_inst_calm", "split": "train", "seed": 107, "instruction": "用平静沉稳的语气说。", "text": "根据目前的记录来看，系统状态非常稳定，可以继续下一步操作。"},
    {"id": "zh_inst_surprised", "split": "train", "seed": 108, "instruction": "用惊讶好奇的语气说。", "text": "咦？你竟然这么快就把全部功能都做完了，怎么做到的呀？"},
    # Train replay: English & code-switched
    {"id": "en_daily_01", "split": "train", "seed": 109, "instruction": "", "text": "Good morning! Everything is ready, so we can start whenever you like."},
    {"id": "en_daily_02", "split": "train", "seed": 110, "instruction": "", "text": "Please double check the configuration file before running the next experiment."},
    {"id": "mix_zh_en_01", "split": "train", "seed": 111, "instruction": "", "text": "这个模型的推论速度很稳定，显存占用也控制得不错。"},
    {"id": "zh_short_01", "split": "train", "seed": 112, "instruction": "", "text": "好的，收到，我马上处理。"},
    # Held-out validation replay (base capability drift anchor)
    {"id": "val_zh_daily", "split": "val", "seed": 201, "instruction": "", "text": "明天早上的会议改到十点开始，大家记得提前几分钟上线。"},
    {"id": "val_zh_inst_happy", "split": "val", "seed": 202, "instruction": "用温柔开心的语气说。", "text": "辛苦啦，忙完这一轮就能好好放松一下，今晚想吃什么呢？"},
    {"id": "val_en_daily", "split": "val", "seed": 203, "instruction": "", "text": "The audio quality sounds very clean and natural after the update."},
    {"id": "val_zh_long_eos", "split": "val", "seed": 204, "instruction": "", "text": "如果遇到不确定的问题，先停下来把目前的观察结果和日志记录清楚，再决定下一个验证步骤，这样既省时间也不容易出错。"},
]


def synthesize_base_clip(
    cli_exe: Path,
    base_gguf: Path,
    prompt_speech_gguf: Path,
    out_wav: Path,
    text: str,
    instruction: str,
    seed: int,
    *,
    backend: str | None = None,
    kv_cache_type: str | None = "f16",
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli_exe),
        "--model", str(base_gguf),
        "--prompt-speech", str(prompt_speech_gguf),
        "--text", text,
        "--output", str(out_wav),
        "--seed", str(seed),
        "--strict-seed", "1",
    ]
    if backend:
        cmd.extend(["--backend", backend])
    if kv_cache_type:
        cmd.extend(["--llm-kv-cache-type", kv_cache_type])
    if instruction:
        cmd.extend(["--mode", "instruct", "--instruction", instruction])
    proc = subprocess.run(
        cmd,
        cwd=str(cli_exe.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0 or not out_wav.is_file() or out_wav.stat().st_size < 1024:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(f"Base replay synthesis failed for {out_wav.name} (rc={proc.returncode}): {tail}")


def extract_speech_tokens(
    cli_exe: Path,
    speech_tokenizer_onnx: Path,
    campplus_onnx: Path,
    wav_path: Path,
    norm_text: str,
    gguf_out: Path,
) -> list[int]:
    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli_exe),
        "--frontend-only",
        "--speech-tokenizer", str(speech_tokenizer_onnx),
        "--campplus", str(campplus_onnx),
        "--prompt-audio", str(wav_path),
        "--prompt-text", norm_text,
        "--prompt-speech-output", str(gguf_out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(cli_exe.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0 or not gguf_out.is_file():
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        raise RuntimeError(f"Frontend token extraction failed for {wav_path.name}: {tail}")
    tensors, data_offset = parse_gguf_tensors(gguf_out)
    raw = gguf_out.read_bytes()
    return load_gguf_numpy(raw, data_offset, tensors["tokens"]).astype("int64").tolist()


def main():
    parser = argparse.ArgumentParser(description="Build reusable CosyVoice3 Base Replay Pack")
    parser.add_argument("--config", type=Path, default=None, help="Optional train_config.json to resolve default paths")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for replay pack")
    parser.add_argument("--base-gguf", type=Path, default=None, help="Shared F16 or mixed HiFT-F32 base GGUF")
    parser.add_argument("--runtime-dir", type=Path, default=None, help="Directory containing cosyvoice-cli & shared lib")
    parser.add_argument("--speech-tokenizer", type=Path, default=None, help="Path to speech_tokenizer_v3.onnx")
    parser.add_argument("--campplus", type=Path, default=None, help="Path to campplus.onnx")
    parser.add_argument(
        "--prompt-speech",
        type=Path,
        default=None,
        help="Reference prompt_speech.gguf used to synthesize base replay clips (required unless --from-dir is given)",
    )
    parser.add_argument("--from-dir", type=Path, default=None, help="Use existing curated WAV+TXT folder instead of synthesizing")
    parser.add_argument("--prompts-json", type=Path, default=None, help="Optional custom JSON list of replay prompts")
    parser.add_argument("--backend", default=None, help="Optional cosyvoice-cli backend (e.g. vulkan0, cuda0, cpu)")
    parser.add_argument("--kv-cache-type", default="f16", help="LLM KV cache type for replay synthesis (default: f16)")
    parser.add_argument("--instruction-prefix", default="You are a helpful assistant.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing replay_manifest.json")
    args = parser.parse_args()

    cfg = {}
    cfg_dir = Path.cwd()
    if args.config is not None:
        cfg_path = args.config.resolve()
        cfg_dir = cfg_path.parent
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    paths = cfg.get("paths", {})
    eval_cfg = cfg.get("eval", {})

    def _req_path(cli_val: Path | None, cfg_key: str, flag_name: str) -> Path:
        if cli_val is not None:
            return cli_val.resolve()
        if paths.get(cfg_key):
            return resolve_cfg_path(paths[cfg_key], cfg_dir)
        parser.error(f"Missing required path: pass {flag_name} or provide paths.{cfg_key} in --config")

    base_gguf = _req_path(args.base_gguf, "infer_base_gguf" if paths.get("infer_base_gguf") else "base_gguf", "--base-gguf")
    runtime_dir = _req_path(args.runtime_dir, "cosyvoice_runtime_dir", "--runtime-dir")
    speech_tokenizer = _req_path(args.speech_tokenizer, "speech_tokenizer_onnx", "--speech-tokenizer")
    campplus = _req_path(args.campplus, "campplus_onnx", "--campplus")
    backend = args.backend if args.backend is not None else eval_cfg.get("backend")

    out_dir = args.out_dir.resolve()
    manifest_path = out_dir / "replay_manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"Replay manifest already exists: {manifest_path} (use --force to rebuild)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"
    gguf_dir = out_dir / "prompt_gguf"
    cli_exe = find_cosyvoice_cli(runtime_dir)
    ffmpeg_exe = find_ffmpeg_binary("ffmpeg", runtime_dir)

    items: list[dict] = []
    if args.from_dir is not None:
        raw_entries = discover_raw_entries(args.from_dir.resolve())
        if not raw_entries:
            raise SystemExit(f"No audio+text pairs found in {args.from_dir}")
        for idx, e in enumerate(raw_entries):
            pcm, sr = load_audio_mono_24k(e["audio_path"], ffmpeg_exe)
            norm_pcm, metrics = normalize_audio_waveform(pcm, sr)
            wav_out = wav_dir / f"{e['id']}.wav"
            write_wav_pcm16(wav_out, norm_pcm, sr)
            split = "val" if (idx % 5 == 4) else "train"
            items.append({
                "id": e["id"],
                "split": split,
                "raw_text": e["raw_text"],
                "norm_text": normalize_transcript_text(e["raw_text"], simplify_chinese=True),
                "instruction": normalize_transcript_text(e.get("instruction", ""), simplify_chinese=True) if e.get("instruction") else "",
                "audio_path": str(wav_out),
                "audio_metrics": metrics,
            })
    else:
        if args.prompt_speech is None:
            if paths.get("processed_dataset_dir"):
                cand_ref = resolve_cfg_path(paths["processed_dataset_dir"], cfg_dir) / "recommended_reference.gguf"
                if cand_ref.is_file():
                    prompt_speech = cand_ref
                else:
                    parser.error("Pass --prompt-speech pointing to a clean reference prompt_speech.gguf (or use --from-dir)")
            else:
                parser.error("Pass --prompt-speech pointing to a clean reference prompt_speech.gguf (or use --from-dir)")
        else:
            prompt_speech = args.prompt_speech.resolve()

        prompt_list = (
            json.loads(args.prompts_json.read_text(encoding="utf-8-sig"))
            if args.prompts_json is not None
            else DEFAULT_REPLAY_PROMPTS
        )
        print(f"[1/3] Synthesizing {len(prompt_list)} base self-distillation replay clips...")
        for idx, p in enumerate(prompt_list, 1):
            raw_wav = out_dir / "raw_synth" / f"{p['id']}.wav"
            synthesize_base_clip(
                cli_exe=cli_exe,
                base_gguf=base_gguf,
                prompt_speech_gguf=prompt_speech,
                out_wav=raw_wav,
                text=p["text"],
                instruction=p.get("instruction", ""),
                seed=int(p.get("seed", 100 + idx)),
                backend=backend,
                kv_cache_type=args.kv_cache_type,
            )
            pcm, sr = load_audio_mono_24k(raw_wav, ffmpeg_exe)
            norm_pcm, metrics = normalize_audio_waveform(pcm, sr)
            wav_out = wav_dir / f"{p['id']}.wav"
            write_wav_pcm16(wav_out, norm_pcm, sr)
            norm_text = normalize_transcript_text(p["text"], simplify_chinese=True)
            norm_inst = normalize_transcript_text(p.get("instruction", ""), simplify_chinese=True) if p.get("instruction") else ""
            items.append({
                "id": p["id"],
                "split": p.get("split", "train"),
                "raw_text": p["text"],
                "norm_text": norm_text,
                "instruction": norm_inst,
                "audio_path": str(wav_out),
                "audio_metrics": metrics,
            })
            print(f"  [{idx}/{len(prompt_list)}] {p['id']} ({p.get('split', 'train')}, {len(norm_pcm)/sr:.2f}s)")

    print("[2/3] Extracting 25 Hz Speech Tokens and BPE Text Tokens...")
    with NativeCosyVoiceTokenizer(runtime_dir, base_gguf) as tokenizer:
        prefix_tokens = tokenizer.encode(f"{args.instruction_prefix}<|endofprompt|>")
        for item in items:
            gguf_out = gguf_dir / f"{item['id']}.gguf"
            sp_tokens = extract_speech_tokens(
                cli_exe,
                speech_tokenizer,
                campplus,
                Path(item["audio_path"]),
                item["norm_text"],
                gguf_out,
            )
            item["prompt_gguf_path"] = str(gguf_out)
            item["speech_tokens"] = sp_tokens
            item["duration_sec"] = round(len(sp_tokens) / 25.0, 3)
            item["text_tokens"] = tokenizer.encode(item["norm_text"])
            item["full_text_tokens"] = tokenizer.encode(
                f"{args.instruction_prefix}<|endofprompt|>{item['norm_text']}"
            )
            if item.get("instruction"):
                item["instruct_prefix_tokens"] = tokenizer.encode(
                    f"{args.instruction_prefix} {item['instruction']}<|endofprompt|>"
                )

    train_replay = [x for x in items if x["split"] != "val"]
    val_replay = [x for x in items if x["split"] == "val"]
    payload = {
        "schema": 1,
        "source": "from_dir" if args.from_dir else "base_self_distillation",
        "base_gguf": str(base_gguf),
        "instruction_prefix": args.instruction_prefix,
        "prefix_tokens": prefix_tokens,
        "train_replay": train_replay,
        "val_replay": val_replay,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3/3] Replay Pack ready: train={len(train_replay)}, val={len(val_replay)} -> {manifest_path}")


if __name__ == "__main__":
    main()
