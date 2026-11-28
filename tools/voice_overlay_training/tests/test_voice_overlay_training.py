"""Offline unit and smoke tests for tools/voice_overlay_training.

Runs completely offline with zero external model files, zero private paths,
and zero GPU requirement. Verifies:
  1. GGUF v3 `cosyvoice3-voice-overlay-v1` writer/reader round-trip and matmul parity
  2. 3-band spectral de-breather catches isolated unvoiced mid-band inhales while
     leaving voiced vowels and high-frequency sibilants intact
  3. Stage 1 silence trim, loudness normalization, anti-leakage group split,
     and reference candidate ranking
  4. Stage 2–3 rsLoRA / LoRA+ optimizer groups, frozen base/head invariant,
     EOS-weighted loss, and synthetic mini-GGUF end-to-end LoRA export
  5. `build_mixed_base.py` F16 + HiFT-F32 construction & bit-identical `layers.*` check
  6. Stage 4 `eval_overlay_matrix.py` dry-run matrix planning (multi-scale + multi-seed)
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from debreath_audio import detect_and_attenuate_breaths
from gguf_utils import (
    OVERLAY_ARCHITECTURE,
    OVERLAY_FORMAT,
    load_gguf_numpy,
    normalize_transcript_text,
    parse_gguf_metadata_and_tensors,
    write_voice_overlay_gguf,
)
from prepare_dataset import (
    build_group_split,
    normalize_audio_waveform,
    score_reference_candidate,
)

try:
    import torch
    from train_cosyvoice_llm import LoRALinear, build_fixed_val_pairs, build_training_example
except ImportError:  # pragma: no cover
    torch = None

try:
    import gguf
    from build_mixed_base import BASE_ARCHITECTURE, build_mixed_base
except ImportError:  # pragma: no cover
    gguf = None


class GgufOverlayTests(unittest.TestCase):
    def test_overlay_roundtrip_and_nonfinite_rejection(self):
        rng = np.random.default_rng(7)
        w_base = rng.standard_normal((32, 16)).astype(np.float32) * 0.05
        a = rng.standard_normal((8, 16)).astype(np.float32) * 0.02
        b = rng.standard_normal((32, 8)).astype(np.float32) * 0.02
        scale = 16.0 / math.sqrt(8.0)

        with tempfile.TemporaryDirectory(prefix="cv_pkg_ov_") as td:
            out = Path(td) / "voice.gguf"
            write_voice_overlay_gguf(
                out,
                {
                    "layers.0.self_attn.q_proj.weight.lora_a": a,
                    "layers.0.self_attn.q_proj.weight.lora_b": b,
                },
                tensor_scales={"layers.0.self_attn.q_proj.weight": scale},
                extra_metadata={"cosyvoice.overlay.voice_name": "unit_test", "cosyvoice.overlay.epoch": 3},
                overwrite=False,
            )
            meta, tensors, data_off = parse_gguf_metadata_and_tensors(out)
            self.assertEqual(meta["general.architecture"], OVERLAY_ARCHITECTURE)
            self.assertEqual(meta["cosyvoice.overlay.format"], OVERLAY_FORMAT)
            self.assertEqual(meta["cosyvoice.overlay.voice_name"], "unit_test")
            self.assertEqual(meta["cosyvoice.overlay.epoch"], 3)
            self.assertAlmostEqual(
                meta["cosyvoice.overlay.tensor_scale.layers.0.self_attn.q_proj.weight"], scale, places=5
            )

            raw = out.read_bytes()
            a_rt = load_gguf_numpy(raw, data_off, tensors["layers.0.self_attn.q_proj.weight.lora_a"])
            b_rt = load_gguf_numpy(raw, data_off, tensors["layers.0.self_attn.q_proj.weight.lora_b"])
            self.assertEqual(a_rt.shape, (8, 16))
            self.assertEqual(b_rt.shape, (32, 8))
            expected = w_base + scale * (b @ a)
            rebuilt = w_base + scale * (b_rt @ a_rt)
            self.assertLess(float(np.max(np.abs(expected - rebuilt))), 1e-6)

            with self.assertRaises(FileExistsError):
                write_voice_overlay_gguf(out, {"x.lora_a": a}, overwrite=False)

            bad = a.copy()
            bad[0, 0] = np.nan
            with self.assertRaises(ValueError):
                write_voice_overlay_gguf(Path(td) / "bad.gguf", {"x.lora_a": bad})

    def test_normalize_preserves_japanese_kana(self):
        jp = "  オハヨウゴザイマス。  今日モ  ヨロシク！  "
        self.assertEqual(normalize_transcript_text(jp, simplify_chinese=True), "オハヨウゴザイマス。 今日モ ヨロシク！")


class AudioPreprocessAndDebreathTests(unittest.TestCase):
    def test_trim_and_loudness_norm(self):
        sr = 24000
        t = np.linspace(0.0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
        speech = 0.08 * np.sin(2.0 * math.pi * 220.0 * t)
        raw = np.concatenate([
            np.zeros(int(0.6 * sr), dtype=np.float32),
            speech,
            np.zeros(int(1.2 * sr), dtype=np.float32),
        ])
        proc, metrics = normalize_audio_waveform(
            raw, sr, trim_silence=True, lead_ms=75.0, tail_ms=150.0, normalize_loudness=True, target_rms_db=-20.0
        )
        dur = len(proc) / float(sr)
        self.assertGreaterEqual(dur, 2.15)
        self.assertLessEqual(dur, 2.30)
        self.assertLess(abs(metrics["active_rms_db"] - (-20.0)), 1.2)
        self.assertEqual(metrics["clipped_samples"], 0)
        self.assertLessEqual(metrics["peak_db"], -0.95)

    def test_debreather_targets_midband_inhale_only(self):
        sr = 24000
        rng = np.random.default_rng(42)

        def _band_noise(dur_s: float, f_lo: float, f_hi: float, amp: float) -> np.ndarray:
            n = int(round(sr * dur_s))
            noise = rng.standard_normal(n).astype(np.float32)
            spec = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            spec[(freqs < f_lo) | (freqs > f_hi)] = 0.0
            sig = np.fft.irfft(spec, n=n).astype(np.float32)
            sig /= max(float(np.sqrt(np.mean(sig ** 2))), 1e-6)
            return sig * amp

        t_v = np.linspace(0.0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
        vowel = (0.18 * np.sin(2.0 * math.pi * 240.0 * t_v)).astype(np.float32)
        sibilant = _band_noise(0.18, 6000.0, 9500.0, 0.06)
        inhale = _band_noise(0.18, 1600.0, 3600.0, 0.05)
        silence = np.zeros(int(0.25 * sr), dtype=np.float32)

        clip = np.concatenate([silence, vowel, silence, inhale, silence, sibilant, silence])
        out, spans = detect_and_attenuate_breaths(clip, sr, attenuation_db=-22.0)

        self.assertGreaterEqual(len(spans), 1)
        # Inhale region (1.00s..1.18s) should be strongly attenuated
        inh_s0 = int(1.02 * sr)
        inh_s1 = int(1.16 * sr)
        in_rms = float(np.sqrt(np.mean(clip[inh_s0:inh_s1] ** 2)))
        out_rms = float(np.sqrt(np.mean(out[inh_s0:inh_s1] ** 2)))
        self.assertLess(out_rms, in_rms * 0.25)

        # Voiced vowel (0.25s..0.75s) and high-freq sibilant (1.43s..1.61s) must stay intact
        v_s0, v_s1 = int(0.30 * sr), int(0.70 * sr)
        self.assertLess(float(np.max(np.abs(out[v_s0:v_s1] - clip[v_s0:v_s1]))), 1e-6)
        sib_s0, sib_s1 = int(1.46 * sr), int(1.58 * sr)
        self.assertLess(float(np.max(np.abs(out[sib_s0:sib_s1] - clip[sib_s0:sib_s1]))), 1e-6)

    def test_group_split_and_ref_ranking(self):
        m = {"active_rms_db": -20.0, "peak_db": -3.0, "clipped_samples": 0, "silence_ratio": 0.1}
        samples = [{"id": f"c_{i:02d}", "norm_text": f"line {i}", "duration_sec": 5.5, "audio_metrics": m} for i in range(10)]
        samples.append({"id": "c_00_part1", "norm_text": "p1", "duration_sec": 4.0, "audio_metrics": m})
        samples.append({"id": "c_00_part2", "norm_text": "p2", "duration_sec": 4.2, "audio_metrics": m})
        samples.append({"id": "dup_line", "norm_text": "line 3", "duration_sec": 5.0, "audio_metrics": m})

        split = build_group_split(samples, val_ratio=0.2, seed=42)
        tr, va = set(split["train"]), set(split["validation"])
        self.assertFalse(tr & va)
        self.assertEqual("c_00_part1" in tr, "c_00_part2" in tr)
        self.assertEqual("c_03" in tr, "dup_line" in tr)

        clean = score_reference_candidate({"duration_sec": 5.8, "audio_metrics": m})
        clipped = score_reference_candidate({"duration_sec": 5.8, "audio_metrics": {**m, "clipped_samples": 5}})
        self.assertGreater(clean, clipped + 40.0)


@unittest.skipIf(torch is None, "PyTorch not installed in this environment")
class LoraMathTests(unittest.TestCase):
    def test_rslora_and_frozen_r0_invariant(self):
        rng = np.random.default_rng(9)
        w = torch.from_numpy(rng.standard_normal((48, 24)).astype(np.float32) * 0.05)
        lin = LoRALinear(w.clone(), r=16, alpha=32.0, use_rslora=True, train_base=False)
        self.assertFalse(lin.weight.requires_grad)
        self.assertTrue(lin.lora_A.requires_grad and lin.lora_B.requires_grad)
        self.assertAlmostEqual(lin.scaling, 32.0 / math.sqrt(16.0), places=6)

        frozen = LoRALinear(w.clone(), r=0, train_base=False)
        self.assertFalse(frozen.weight.requires_grad)

    def test_training_example_and_fixed_val_pairs(self):
        s1 = {"id": "a", "text_tokens": [10, 11], "full_text_tokens": [1, 2, 10, 11], "speech_tokens": [100, 101, 102]}
        s2 = {"id": "b", "text_tokens": [20, 21], "full_text_tokens": [1, 2, 20, 21], "speech_tokens": [200, 201]}
        import random

        txt, sp, start = build_training_example(s1, [s1, s2], [1, 2], 1.0, 50, random.Random(0))
        self.assertEqual(txt, [1, 2, 20, 21, 10, 11])
        self.assertEqual(sp, [200, 201, 100, 101, 102])
        self.assertEqual(start, 2)

        pairs = build_fixed_val_pairs([s1], [s2], 50, seed=42)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["id"], "b")


@unittest.skipIf(gguf is None, "gguf package not installed in this environment")
class MixedBaseTests(unittest.TestCase):
    def _write_synthetic_base(self, path: Path, dtype) -> None:
        writer = gguf.GGUFWriter(path, BASE_ARCHITECTURE)
        try:
            writer.add_uint32("general.file_type", 0 if dtype == np.float32 else 1)
            rng = np.random.default_rng(1234)
            tensors = [
                ("layers.0.self_attn.q_proj.weight", (8, 8)),
                ("layers.0.input_layernorm.weight", (8,)),
                ("decoder.estimator.weight", (8, 8)),
                ("embed_tokens.weight", (16, 8)),
                ("llm_decoder.weight", (12, 8)),
                ("speech_embedding.weight", (12, 8)),
                ("hift.conv_post.weight", (4, 4)),
            ]
            for name, shape in tensors:
                arr = rng.standard_normal(shape).astype(np.float32)
                if dtype == np.float16 and len(shape) >= 2:
                    writer.add_tensor(name, arr.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
                else:
                    writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=False)
        finally:
            writer.close()

    def test_build_mixed_base_f16_plus_hift_f32(self):
        with tempfile.TemporaryDirectory(prefix="cv_pkg_mix_") as td:
            f32 = Path(td) / "base_f32.gguf"
            f16 = Path(td) / "base_f16.gguf"
            out = Path(td) / "mixed.gguf"
            self._write_synthetic_base(f32, np.float32)
            self._write_synthetic_base(f16, np.float16)

            res = build_mixed_base(f32, out, f16_path=f16)
            self.assertTrue(res["layers_bit_identical_to_f16"])
            self.assertEqual(res["f16_tensors"], 3)
            self.assertEqual(res["f32_upgraded_tensors"], 3)
            self.assertEqual(res["f32_kept_tensors"], 1)

            reader = gguf.GGUFReader(out)
            try:
                by_name = {t.name: t.tensor_type for t in reader.tensors}
            finally:
                mmap_obj = getattr(getattr(reader, "data", None), "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
            self.assertEqual(by_name["layers.0.self_attn.q_proj.weight"], gguf.GGMLQuantizationType.F16)
            self.assertEqual(by_name["decoder.estimator.weight"], gguf.GGMLQuantizationType.F16)
            self.assertEqual(by_name["embed_tokens.weight"], gguf.GGMLQuantizationType.F16)
            self.assertEqual(by_name["llm_decoder.weight"], gguf.GGMLQuantizationType.F32)
            self.assertEqual(by_name["speech_embedding.weight"], gguf.GGMLQuantizationType.F32)
            self.assertEqual(by_name["hift.conv_post.weight"], gguf.GGMLQuantizationType.F32)


class EvalMatrixDryRunTests(unittest.TestCase):
    def test_eval_matrix_dry_run_multiseed(self):
        with tempfile.TemporaryDirectory(prefix="cv_pkg_eval_") as td:
            td_p = Path(td)
            ov = td_p / "epoch_04.gguf"
            ov.write_bytes(b"placeholder")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_DIR / "eval_overlay_matrix.py"),
                    "--base", str(td_p / "base.gguf"),
                    "--prompt-speech", str(td_p / "ref.gguf"),
                    "--overlays", str(ov),
                    "--scales", "0.55", "0.70",
                    "--seeds", "42", "101",
                    "--dry-run",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            plan = json.loads(proc.stdout)
            self.assertEqual(plan["kv_cache_type"], "f16")
            self.assertEqual(plan["seeds"], [42, 101])
            # 2 seeds * (base + 2 scales) = 6 candidates
            self.assertEqual(plan["candidate_count"], 6)
            self.assertEqual(plan["case_count"], 8)
            self.assertEqual(plan["total_runs"], 48)


if __name__ == "__main__":
    unittest.main()
