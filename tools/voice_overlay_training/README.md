# CosyVoice3 Voice Overlay LoRA Training Toolkit

Standalone, cross-platform Python toolchain for fine-tuning the `CosyVoice3-2512` Speech-Token LLM with **Rank-16 rsLoRA + LoRA+** and exporting lightweight native `cosyvoice3-voice-overlay-v1` GGUF files (`~33.6 MB` in F32) for `cosyvoice.cpp`.

For the full training methodology, hyperparameter rationale, dataset hygiene checklist, and listening acceptance rubric, see [`docs/VOICE_OVERLAY_TRAINING_zh.md`](../../docs/VOICE_OVERLAY_TRAINING_zh.md).

## Files

| File | Stage | Purpose |
| :--- | :---: | :--- |
| `build_mixed_base.py` | Stage 0 | Builds a quality-oriented shared base GGUF (`F16` LLM/DiT/embed_tokens + native `F32` HiFT/F0/head/norms, bit-identical `layers.*` to official F16) |
| `debreath_audio.py` | Pre-Stage 1 | Optional zero-dependency 3-band spectral de-breather for close-mic inter-clause inhales (preserves vowels & sibilants; supports `--dry-run`) |
| `prepare_dataset.py` | Stage 1 | Silence trimming, `-20 dBFS` loudness normalization, native `cosyvoice-cli --frontend-only` token extraction, group anti-leakage `split.json`, and Top-5 reference ranking |
| `build_replay_pack.py` | Stage 1 | Builds a reusable multilingual + instruction Base Self-Distillation Replay Pack to prevent catastrophic forgetting |
| `train_cosyvoice_llm.py` | Stages 2–3 | Pure-FP32 Speech-LLM LoRA trainer (rsLoRA, LoRA+, lower-layer LR protection, 3-mode + Replay training, EOS up-weighting, direct `cosyvoice3-voice-overlay-v1` export) |
| `eval_overlay_matrix.py` | Stage 4 | Free-generation multilingual/instruct/long-EOS evaluation sweep across `(epoch, scale, seeds)` with Markdown listening report |
| `gguf_utils.py` | Shared | Minimal GGUF v3 parser/writer, cross-platform `cosyvoice-cli` / shared-library discovery, and text normalization helpers |
| `train_config.example.json` | Config | Ready-to-edit configuration template with verified v2 defaults |
| `tests/test_voice_overlay_training.py` | Tests | Offline unit & smoke test suite (no GPU or external model files required) |

## Quickstart

```bash
# 1. Install dependencies (ensure PyTorch matches your CUDA/ROCm platform)
python -m pip install -r tools/voice_overlay_training/requirements.txt

# 2. (Optional) Build mixed-precision shared base GGUF (+~65 MB over stock F16)
python tools/voice_overlay_training/build_mixed_base.py \
  --f32 ./models/CosyVoice3-2512_F32.gguf \
  --f16 ./models/CosyVoice3-2512_F16.gguf \
  --out ./models/CosyVoice3-2512_F16_HiFT-F32.gguf

# 3. Copy & edit config
cp tools/voice_overlay_training/train_config.example.json ./train_config.json

# 4. Prepare dataset & reusable base replay pack
python tools/voice_overlay_training/prepare_dataset.py --config ./train_config.json
python tools/voice_overlay_training/build_replay_pack.py \
  --config ./train_config.json --out-dir ./data/replay

# 5. Validate setup, then run LoRA training
python tools/voice_overlay_training/train_cosyvoice_llm.py --config ./train_config.json --dry-run-check
python tools/voice_overlay_training/train_cosyvoice_llm.py --config ./train_config.json --start-training

# 6. Run Stage 4 evaluation matrix (with F16 KV cache) and audition listening_report.md
#    Tip: Recommend using Gemini 4 (native multimodal audio) to audition the generated WAVs
#    and shortlist the top 1–2 (epoch, scale) candidates before final human check.
python tools/voice_overlay_training/eval_overlay_matrix.py \
  --config ./train_config.json --scales 0.55 0.70 0.85
```

## Running Tests

```bash
python -m unittest discover -s tools/voice_overlay_training/tests -p "test_*.py" -v
```
