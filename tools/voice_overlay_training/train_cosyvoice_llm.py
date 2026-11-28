"""Stages 2–3: Pure-FP32 Speech-LLM LoRA Trainer for CosyVoice3-2512 (`cosyvoice3-voice-overlay-v1`).

Trains low-rank adapters directly against the 24-layer Qwen2-0.5B Speech-Token LLM
inside an official `CosyVoice3-2512` F32 (or F16) GGUF while freezing Flow/DiT, HiFT,
`embed_tokens`, and (by default) `speech_embedding` + `llm_decoder`.

Key stability mechanisms for small single-speaker corpora (30–200 clips):
  - rsLoRA (`alpha / sqrt(r)`) + LoRA+ asymmetric `B` learning rate
  - Lower-layer protection (`L0..L7` at `0.25x` LR) to preserve multilingual alignment
  - 3-mode mixed training (Zero-Shot paired continuation + Single-utterance + Instruct)
  - Optional Base Replay Regularization with held-out `val_replay_ce` drift monitoring
  - Label smoothing + EOS stop-token loss up-weighting (`1.5x`)
  - Direct atomic export to lightweight native `cosyvoice3-voice-overlay-v1` GGUF (~33.6 MB @ r=16 F32)
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gguf_utils import (
    NativeCosyVoiceTokenizer,
    load_gguf_numpy,
    parse_gguf_tensors,
    write_voice_overlay_gguf,
)


def resolve_cfg_path(raw: str | Path, cfg_dir: Path) -> Path:
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (cfg_dir / p).resolve()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    out = x_f32 * torch.rsqrt(variance + eps) * weight.float()
    return out.to(orig_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class LoRALinear(nn.Module):
    """Linear layer with optional LoRA / rsLoRA adapter exporting cleanly to Voice Overlay GGUF."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        r: int = 0,
        alpha: float = 16.0,
        dropout: float = 0.0,
        *,
        use_rslora: bool = False,
        train_base: bool = False,
    ):
        super().__init__()
        out_features, in_features = weight.shape
        self.weight = nn.Parameter(weight, requires_grad=train_base)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=train_base)
        else:
            self.register_parameter("bias", None)

        self.r = int(r)
        self.alpha = float(alpha)
        self.use_rslora = bool(use_rslora)
        if self.r > 0:
            denom = math.sqrt(self.r) if self.use_rslora else float(self.r)
            self.scaling = self.alpha / denom
            self.lora_A = nn.Parameter(torch.empty((self.r, in_features), dtype=torch.float32, device=weight.device))
            self.lora_B = nn.Parameter(torch.zeros((out_features, self.r), dtype=torch.float32, device=weight.device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.scaling = 0.0
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
        if self.r > 0:
            lx = self.drop(x.float())
            lora_out = (lx @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return base_out + lora_out.to(x.dtype)
        return base_out

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        w = self.weight.float()
        if self.r > 0:
            w = w + (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        return w


class CosyVoiceAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        loader,
        r: int,
        alpha: float,
        dropout: float,
        *,
        use_rslora: bool = False,
        full_ft: bool = False,
    ):
        super().__init__()
        p = f"layers.{layer_idx}.self_attn."
        self.q_proj = LoRALinear(loader(p + "q_proj.weight"), loader(p + "q_proj.bias"), r, alpha, dropout, use_rslora=use_rslora, train_base=full_ft)
        self.k_proj = LoRALinear(loader(p + "k_proj.weight"), loader(p + "k_proj.bias"), r, alpha, dropout, use_rslora=use_rslora, train_base=full_ft)
        self.v_proj = LoRALinear(loader(p + "v_proj.weight"), loader(p + "v_proj.bias"), r, alpha, dropout, use_rslora=use_rslora, train_base=full_ft)
        self.o_proj = LoRALinear(loader(p + "o_proj.weight"), None, r, alpha, dropout, use_rslora=use_rslora, train_base=full_ft)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, 14, 64).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, 2, 64).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, 2, 64).transpose(1, 2)

        q = ((q.float() * cos) + (rotate_half(q.float()) * sin)).to(x.dtype)
        k = ((k.float() * cos) + (rotate_half(k.float()) * sin)).to(x.dtype)

        k = k.repeat_interleave(7, dim=1)
        v = v.repeat_interleave(7, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, 896)
        return self.o_proj(attn_out)


class CosyVoiceMLP(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        loader,
        r: int,
        alpha: float,
        dropout: float,
        *,
        use_rslora: bool = False,
        full_ft: bool = False,
    ):
        super().__init__()
        p = f"layers.{layer_idx}.mlp."
        self.gate_proj = LoRALinear(loader(p + "gate_proj.weight"), None, r, alpha, dropout, use_rslora=use_rslora, train_base=full_ft)
        self.up_proj = LoRALinear(loader(p + "up_proj.weight"), None, r, alpha, dropout, use_rslora=use_rslora, train_base=full_ft)
        self.down_proj = LoRALinear(loader(p + "down_proj.weight"), None, r, alpha, dropout, use_rslora=use_rslora, train_base=full_ft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CosyVoiceLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        loader,
        r: int,
        alpha: float,
        dropout: float,
        full_ft: bool,
        *,
        use_rslora: bool = False,
    ):
        super().__init__()
        p = f"layers.{layer_idx}."
        self.input_layernorm = nn.Parameter(loader(p + "input_layernorm.weight", torch.float32), requires_grad=full_ft)
        self.post_attention_layernorm = nn.Parameter(
            loader(p + "post_attention_layernorm.weight", torch.float32), requires_grad=full_ft
        )
        self.self_attn = CosyVoiceAttention(layer_idx, loader, r, alpha, dropout, use_rslora=use_rslora, full_ft=full_ft)
        self.mlp = CosyVoiceMLP(layer_idx, loader, r, alpha, dropout, use_rslora=use_rslora, full_ft=full_ft)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(rms_norm(x, self.input_layernorm), cos, sin)
        x = x + self.mlp(rms_norm(x, self.post_attention_layernorm))
        return x


class CosyVoice3LLM(nn.Module):
    def __init__(
        self,
        gguf_path: str | Path,
        device: str = "cuda",
        mode: str = "lora",
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        train_speech_head: bool = False,
        *,
        use_rslora: bool = True,
        lower_layers_end: int = 8,
        lower_layers_lr_mult: float = 0.25,
        llm_decoder_lora_r: int | None = 0,
        num_layers: int = 24,
    ):
        super().__init__()
        self.device_name = device
        self.mode = mode
        self.num_layers = num_layers
        self.lower_layers_end = max(0, min(num_layers, int(lower_layers_end)))
        self.lower_layers_lr_mult = float(lower_layers_lr_mult)
        full_ft = mode == "full"

        if llm_decoder_lora_r is not None:
            dec_r = int(llm_decoder_lora_r) if mode == "lora" else 0
        else:
            dec_r = int(lora_r) if (mode == "lora" and train_speech_head) else 0
        dec_alpha = float(lora_alpha) * (dec_r / max(1, int(lora_r))) if dec_r > 0 else float(lora_alpha)

        tensors, data_offset = parse_gguf_tensors(gguf_path)
        with open(gguf_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                def loader(name: str, dtype=torch.float32) -> torch.Tensor:
                    arr = load_gguf_numpy(mm, data_offset, tensors[name])
                    return torch.from_numpy(arr).to(device=device, dtype=dtype)

                self.embed_tokens = nn.Parameter(loader("embed_tokens.weight", torch.float32), requires_grad=False)
                self.speech_embedding = nn.Parameter(
                    loader("speech_embedding.weight", torch.float32),
                    requires_grad=(full_ft or train_speech_head),
                )
                self.llm_decoder = LoRALinear(
                    loader("llm_decoder.weight", torch.float32),
                    None,
                    r=dec_r,
                    alpha=dec_alpha,
                    dropout=lora_dropout,
                    use_rslora=use_rslora,
                    train_base=full_ft,
                )
                self.norm = nn.Parameter(loader("norm.weight", torch.float32), requires_grad=full_ft)

                layers = []
                for i in range(num_layers):
                    if mode == "lora":
                        layer_r = 0 if (i < self.lower_layers_end and self.lower_layers_lr_mult == 0.0) else int(lora_r)
                    else:
                        layer_r = 0
                    layers.append(
                        CosyVoiceLayer(
                            i,
                            loader,
                            r=layer_r,
                            alpha=lora_alpha,
                            dropout=lora_dropout,
                            full_ft=full_ft,
                            use_rslora=use_rslora,
                        )
                    )
                self.layers = nn.ModuleList(layers)
            finally:
                mm.close()

        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64.0))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward_loss(
        self,
        text_token_ids: list[int],
        speech_token_ids: list[int],
        loss_start_speech_idx: int = 0,
        sos_token_id: int = 6561,
        eos_token_id: int = 6562,
        task_token_id: int = 6563,
        *,
        label_smoothing: float = 0.0,
        eos_loss_weight: float = 1.0,
        return_components: bool = False,
    ):
        device = self.inv_freq.device
        sos_emb = self.speech_embedding[sos_token_id : sos_token_id + 1]
        task_emb = self.speech_embedding[task_token_id : task_token_id + 1]
        txt_ids = torch.tensor(text_token_ids, device=device, dtype=torch.long)
        sp_ids = torch.tensor(speech_token_ids, device=device, dtype=torch.long)

        txt_emb = self.embed_tokens[txt_ids].float()
        sp_emb = self.speech_embedding[sp_ids]

        x = torch.cat([sos_emb, txt_emb, task_emb, sp_emb], dim=0).unsqueeze(0)
        seq_len = x.shape[1]
        cos, sin = self._rope(seq_len)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = rms_norm(x, self.norm)
        task_idx = 1 + len(text_token_ids)
        start_pos = task_idx + loss_start_speech_idx
        pred_hidden = x[0, start_pos:]
        logits = self.llm_decoder(pred_hidden)

        targets = speech_token_ids[loss_start_speech_idx:] + [eos_token_id]
        target_tensor = torch.tensor(targets, device=device, dtype=torch.long)

        if not return_components and label_smoothing == 0.0 and eos_loss_weight == 1.0:
            return F.cross_entropy(logits.float(), target_tensor)

        per_tok = F.cross_entropy(
            logits.float(),
            target_tensor,
            reduction="none",
            label_smoothing=float(label_smoothing),
        )
        if eos_loss_weight != 1.0:
            weights = torch.ones_like(per_tok)
            weights[-1] = float(eos_loss_weight)
            total_loss = (per_tok * weights).sum() / weights.sum()
        else:
            total_loss = per_tok.mean()

        if return_components:
            with torch.no_grad():
                raw_per_tok = per_tok if label_smoothing == 0.0 else F.cross_entropy(
                    logits.float(), target_tensor, reduction="none"
                )
                raw_ce = float(raw_per_tok.mean().item())
                eos_nll = float(raw_per_tok[-1].item())
            return total_loss, raw_ce, eos_nll
        return total_loss

    def iter_named_lora_modules(self):
        """Yield `(gguf_tensor_name, LoRALinear)` for every adapted linear projection."""
        if self.llm_decoder.r > 0:
            yield "llm_decoder.weight", self.llm_decoder
        for i, layer in enumerate(self.layers):
            p = f"layers.{i}."
            for sub_name, mod in (
                ("self_attn.q_proj.weight", layer.self_attn.q_proj),
                ("self_attn.k_proj.weight", layer.self_attn.k_proj),
                ("self_attn.v_proj.weight", layer.self_attn.v_proj),
                ("self_attn.o_proj.weight", layer.self_attn.o_proj),
                ("mlp.gate_proj.weight", layer.mlp.gate_proj),
                ("mlp.up_proj.weight", layer.mlp.up_proj),
                ("mlp.down_proj.weight", layer.mlp.down_proj),
            ):
                if mod.r > 0:
                    yield p + sub_name, mod

    @torch.no_grad()
    def export_voice_overlay_gguf(
        self,
        out_path: str | Path,
        *,
        precision: str = "f32",
        default_scale: float = 1.0,
        extra_metadata: dict[str, str | float | int] | None = None,
    ) -> Path:
        """Export LoRA factors directly as `cosyvoice3-voice-overlay-v1` GGUF."""
        if self.mode != "lora":
            raise ValueError("export_voice_overlay_gguf is for LoRA mode; use export_gguf_tensor_dict for full SFT")
        dtype = np.float32 if precision == "f32" else np.float16
        tensors: dict[str, np.ndarray] = {}
        tensor_scales: dict[str, float] = {}

        for base_name, mod in self.iter_named_lora_modules():
            a = np.ascontiguousarray(mod.lora_A.detach().float().cpu().numpy(), dtype=dtype)
            b = np.ascontiguousarray(mod.lora_B.detach().float().cpu().numpy(), dtype=dtype)
            tensors[base_name + ".lora_a"] = a
            tensors[base_name + ".lora_b"] = b
            tensor_scales[base_name] = float(mod.scaling)

        if self.speech_embedding.requires_grad:
            tensors["speech_embedding.weight"] = np.ascontiguousarray(
                self.speech_embedding.detach().float().cpu().numpy(), dtype=np.float16
            )

        return write_voice_overlay_gguf(
            out_path,
            tensors,
            tensor_scales=tensor_scales,
            default_scale=default_scale,
            extra_metadata=extra_metadata,
            overwrite=True,
        )

    @torch.no_grad()
    def export_gguf_tensor_dict(self, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """Export merged tensors with exact CosyVoice3-2512 GGUF tensor names."""
        out = {
            "embed_tokens.weight": self.embed_tokens.detach().to(dtype).cpu(),
            "speech_embedding.weight": self.speech_embedding.detach().to(dtype).cpu(),
            "llm_decoder.weight": self.llm_decoder.merged_weight().to(dtype).cpu(),
            "norm.weight": self.norm.detach().to(torch.float32).cpu(),
        }
        for i, layer in enumerate(self.layers):
            p = f"layers.{i}."
            out[p + "input_layernorm.weight"] = layer.input_layernorm.detach().to(torch.float32).cpu()
            out[p + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.detach().to(torch.float32).cpu()
            out[p + "self_attn.q_proj.weight"] = layer.self_attn.q_proj.merged_weight().to(dtype).cpu()
            out[p + "self_attn.q_proj.bias"] = layer.self_attn.q_proj.bias.detach().to(torch.float32).cpu()
            out[p + "self_attn.k_proj.weight"] = layer.self_attn.k_proj.merged_weight().to(dtype).cpu()
            out[p + "self_attn.k_proj.bias"] = layer.self_attn.k_proj.bias.detach().to(torch.float32).cpu()
            out[p + "self_attn.v_proj.weight"] = layer.self_attn.v_proj.merged_weight().to(dtype).cpu()
            out[p + "self_attn.v_proj.bias"] = layer.self_attn.v_proj.bias.detach().to(torch.float32).cpu()
            out[p + "self_attn.o_proj.weight"] = layer.self_attn.o_proj.merged_weight().to(dtype).cpu()
            out[p + "mlp.gate_proj.weight"] = layer.mlp.gate_proj.merged_weight().to(dtype).cpu()
            out[p + "mlp.up_proj.weight"] = layer.mlp.up_proj.merged_weight().to(dtype).cpu()
            out[p + "mlp.down_proj.weight"] = layer.mlp.down_proj.merged_weight().to(dtype).cpu()
        return out


def build_optimizer_param_groups(
    model: CosyVoice3LLM,
    *,
    lr: float,
    weight_decay: float,
    lora_plus_ratio: float = 1.0,
    lower_layers_end: int = 0,
    lower_layers_lr_mult: float = 1.0,
    head_lr_mult: float = 0.25,
) -> list[dict]:
    """Construct AdamW parameter groups supporting LoRA+ (higher LR on B) and lower-layer protection."""
    if model.mode == "full":
        return [{"name": "full_ft", "params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]

    buckets: dict[str, dict] = {}

    def _add(group_name: str, param: nn.Parameter, group_lr: float, group_wd: float):
        if group_name not in buckets:
            buckets[group_name] = {"name": group_name, "params": [], "lr": group_lr, "weight_decay": group_wd}
        buckets[group_name]["params"].append(param)

    layer_re = re.compile(r"^layers\.(\d+)\.")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "speech_embedding" in name:
            _add("speech_embedding", param, lr * head_lr_mult, 0.0)
            continue

        is_b = name.endswith(".lora_B")
        plus_mult = float(lora_plus_ratio) if is_b else 1.0
        ab_tag = "B" if is_b else "A"

        if name.startswith("llm_decoder."):
            _add(f"head_lora_{ab_tag}", param, lr * head_lr_mult * plus_mult, weight_decay)
            continue

        m = layer_re.match(name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx < lower_layers_end:
                _add(f"lower_lora_{ab_tag}", param, lr * lower_layers_lr_mult * plus_mult, weight_decay)
            else:
                _add(f"upper_lora_{ab_tag}", param, lr * plus_mult, weight_decay)
        else:
            _add(f"other_lora_{ab_tag}", param, lr * plus_mult, weight_decay)

    return list(buckets.values())


def build_training_example(
    sample: dict,
    pair_pool: list[dict],
    prefix_tokens: list[int],
    zero_shot_pair_prob: float,
    max_paired_speech_tokens: int,
    rng: random.Random,
    *,
    instruct_prob: float = 0.0,
    neutral_instruct_prefixes: list[list[int]] | None = None,
    is_replay: bool = False,
) -> tuple[list[int], list[int], int]:
    """Return `(text_token_ids, speech_token_ids, loss_start_speech_idx)`."""
    if is_replay and sample.get("instruct_prefix_tokens"):
        return sample["instruct_prefix_tokens"] + sample["text_tokens"], sample["speech_tokens"], 0

    u = rng.random()
    if len(pair_pool) >= 2 and u < zero_shot_pair_prob:
        candidates = [
            s for s in pair_pool
            if s["id"] != sample["id"]
            and (len(s["speech_tokens"]) + len(sample["speech_tokens"])) <= max_paired_speech_tokens
        ]
        if candidates:
            ref = rng.choice(candidates)
            comb_text = prefix_tokens + ref["text_tokens"] + sample["text_tokens"]
            comb_speech = ref["speech_tokens"] + sample["speech_tokens"]
            return comb_text, comb_speech, len(ref["speech_tokens"])

    if not is_replay and u < (zero_shot_pair_prob + instruct_prob):
        if sample.get("instruct_prefix_tokens"):
            return sample["instruct_prefix_tokens"] + sample["text_tokens"], sample["speech_tokens"], 0
        if neutral_instruct_prefixes:
            pref = rng.choice(neutral_instruct_prefixes)
            return pref + sample["text_tokens"], sample["speech_tokens"], 0

    return sample["full_text_tokens"], sample["speech_tokens"], 0


def build_fixed_val_pairs(
    val_samples: list[dict],
    ref_pool: list[dict],
    max_paired_speech_tokens: int,
    seed: int,
) -> list[tuple[dict, dict]]:
    """Build deterministic `(ref_sample, val_sample)` pairs for stable Zero-Shot validation curves."""
    rng = random.Random(seed + 7919)
    pairs: list[tuple[dict, dict]] = []
    for vs in val_samples:
        cands = [
            r for r in ref_pool
            if r["id"] != vs["id"]
            and (len(r["speech_tokens"]) + len(vs["speech_tokens"])) <= max_paired_speech_tokens
        ]
        if cands:
            cands.sort(key=lambda r: abs(len(r["speech_tokens"]) / 25.0 - 5.5))
            top = cands[: min(5, len(cands))]
            pairs.append((rng.choice(top), vs))
    return pairs


@torch.no_grad()
def evaluate_suite(
    model: CosyVoice3LLM,
    val_samples: list[dict],
    val_pairs: list[tuple[dict, dict]],
    prefix_tokens: list[int],
    val_replay: list[dict] | None = None,
) -> dict[str, float | None]:
    """Evaluate 4 validation signals with raw (unsmoothed) CE: single, zero-shot, replay, and EOS NLL."""
    model.eval()
    single_ces, single_eos = [], []
    for s in val_samples:
        _, raw_ce, eos_nll = model.forward_loss(
            s["full_text_tokens"], s["speech_tokens"], loss_start_speech_idx=0, return_components=True
        )
        single_ces.append(raw_ce)
        single_eos.append(eos_nll)

    zs_ces, zs_eos = [], []
    for ref, vs in val_pairs:
        comb_text = prefix_tokens + ref["text_tokens"] + vs["text_tokens"]
        comb_speech = ref["speech_tokens"] + vs["speech_tokens"]
        _, raw_ce, eos_nll = model.forward_loss(
            comb_text, comb_speech, loss_start_speech_idx=len(ref["speech_tokens"]), return_components=True
        )
        zs_ces.append(raw_ce)
        zs_eos.append(eos_nll)

    replay_ces, replay_eos = [], []
    if val_replay:
        for rs in val_replay:
            txt = (rs["instruct_prefix_tokens"] + rs["text_tokens"]) if rs.get("instruct_prefix_tokens") else rs["full_text_tokens"]
            _, raw_ce, eos_nll = model.forward_loss(
                txt, rs["speech_tokens"], loss_start_speech_idx=0, return_components=True
            )
            replay_ces.append(raw_ce)
            replay_eos.append(eos_nll)

    model.train()
    mean = lambda arr: round(sum(arr) / len(arr), 5) if arr else None
    all_eos = single_eos + zs_eos
    return {
        "val_single_ce": mean(single_ces),
        "val_zeroshot_ce": mean(zs_ces) if zs_ces else mean(single_ces),
        "val_replay_ce": mean(replay_ces),
        "val_eos_loss": mean(all_eos),
        "val_single_eos": mean(single_eos),
        "val_zeroshot_eos": mean(zs_eos),
        "val_replay_eos": mean(replay_eos),
    }


def resolve_split(samples: list[dict], proc_dir: Path, val_ratio: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    """Use frozen `split.json` if present and matching; otherwise build deterministic group-aware split."""
    by_id = {s["id"]: s for s in samples}
    split_path = proc_dir / "split.json"
    if split_path.is_file():
        frozen = json.loads(split_path.read_text(encoding="utf-8-sig"))
        train_ids = frozen.get("train", [])
        val_ids = frozen.get("validation", [])
        if set(train_ids) | set(val_ids) == set(by_id) and not (set(train_ids) & set(val_ids)):
            return [by_id[i] for i in train_ids], [by_id[i] for i in val_ids], frozen

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
    target = max(1, int(round(len(samples) * val_ratio))) if len(samples) >= 4 else 0
    val_samples, train_samples = [], []
    for key in keys:
        (val_samples if len(val_samples) < target else train_samples).extend(groups[key])
    if not train_samples:
        train_samples = list(val_samples)
        val_samples = val_samples[:1]
    split_obj = {
        "seed": seed,
        "train": [s["id"] for s in train_samples],
        "validation": [s["id"] for s in val_samples],
        "group_by": "source_parts_and_identical_normalized_text",
    }
    return train_samples, val_samples, split_obj


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CosyVoice3-2512 Speech LLM (Voice Overlay LoRA)")
    parser.add_argument("--config", type=Path, required=True, help="Path to train_config.json")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max optimizer steps for smoke testing")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Compute device")
    parser.add_argument("--start-training", action="store_true", help="Explicitly start training (safety gate)")
    parser.add_argument("--dry-run-check", action="store_true", help="Validate config, data, and split without training")
    parser.add_argument("--allow-cpu-smoke", action="store_true", help="Allow CPU execution when no CUDA/ROCm GPU is present")
    parser.add_argument("--overwrite-run", action="store_true", help="Allow writing into an existing run directory")
    args = parser.parse_args()

    if not args.start_training and not args.dry_run_check:
        parser.error("Safety gate: pass --start-training to run training, or --dry-run-check to validate setup.")

    cfg_path = args.config.resolve()
    cfg_dir = cfg_path.parent
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    paths = cfg["paths"]
    tr_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]

    seed = int(tr_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    proc_dir = resolve_cfg_path(paths["processed_dataset_dir"], cfg_dir)
    base_gguf = resolve_cfg_path(paths["base_gguf"], cfg_dir)
    runs_root = resolve_cfg_path(paths["runs_dir"], cfg_dir)

    manifest_path = proc_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; run prepare_dataset.py first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    samples = manifest["samples"]
    prefix_tokens = manifest["prefix_tokens"]
    if not samples:
        raise ValueError("dataset_manifest.json contains zero valid samples")

    val_ratio = float(ds_cfg.get("val_ratio", 0.1))
    train_samples, val_samples, split_obj = resolve_split(samples, proc_dir, val_ratio=val_ratio, seed=seed)

    replay_train: list[dict] = []
    replay_val: list[dict] = []
    replay_manifest_raw = paths.get("replay_manifest")
    replay_ratio = float(tr_cfg.get("replay_ratio", 0.0))
    if replay_manifest_raw:
        replay_path = resolve_cfg_path(replay_manifest_raw, cfg_dir)
        if not replay_path.is_file():
            raise FileNotFoundError(f"Config specifies replay_manifest={replay_path}, but file does not exist")
        rm = json.loads(replay_path.read_text(encoding="utf-8-sig"))
        replay_train = list(rm.get("train_replay", []))
        replay_val = list(rm.get("val_replay", []))
    elif replay_ratio > 0.0:
        raise ValueError("training.replay_ratio > 0 requires paths.replay_manifest")

    mode = tr_cfg.get("mode", "lora").lower()
    if mode not in ("lora", "full"):
        raise ValueError(f"Unsupported training.mode: {mode}")

    epochs = int(args.epochs if args.epochs is not None else tr_cfg.get("epochs", 6))
    grad_accum = max(1, int(tr_cfg.get("grad_accum_steps", 4)))
    lr = float(tr_cfg.get("learning_rate", 2e-5) if mode == "lora" else tr_cfg.get("full_finetune_lr", 1e-5))
    weight_decay = float(tr_cfg.get("weight_decay", 0.01))
    max_grad_norm = float(tr_cfg.get("max_grad_norm", 1.0))
    pair_prob = float(tr_cfg.get("zero_shot_pair_prob", 0.6))
    instruct_prob = float(tr_cfg.get("instruct_prob", 0.15))
    max_paired_tokens = int(tr_cfg.get("max_paired_speech_tokens", 750))
    label_smoothing = float(tr_cfg.get("label_smoothing", 0.05))
    eos_loss_weight = float(tr_cfg.get("eos_loss_weight", 1.5))
    use_rslora = bool(tr_cfg.get("use_rslora", True))
    lora_plus_ratio = float(tr_cfg.get("lora_plus_ratio", 2.0))
    lower_layers_end = int(tr_cfg.get("lower_layers_end", 8))
    lower_layers_lr_mult = float(tr_cfg.get("lower_layers_lr_mult", 0.25))
    head_lr_mult = float(tr_cfg.get("head_lr_mult", 0.25))
    train_speech_head = bool(tr_cfg.get("train_speech_head", False))
    llm_decoder_lora_r = tr_cfg.get("llm_decoder_lora_r", 0)
    save_every_epochs = max(1, int(tr_cfg.get("save_every_epochs", 1)))
    overlay_precision = str(tr_cfg.get("overlay_precision", "f32")).lower()
    if overlay_precision not in ("f32", "f16"):
        raise ValueError("training.overlay_precision must be 'f32' or 'f16'")
    save_merged_pt = bool(tr_cfg.get("save_merged_pt", False))
    selection_metric = str(tr_cfg.get("selection_metric", "val_zeroshot_ce"))
    replay_drift_warn = float(tr_cfg.get("replay_ce_drift_warn", 0.15))

    if not (0.0 <= label_smoothing < 0.5):
        raise ValueError(f"label_smoothing out of safe range [0, 0.5): {label_smoothing}")
    if not (0.5 <= eos_loss_weight <= 5.0):
        raise ValueError(f"eos_loss_weight out of safe range [0.5, 5.0]: {eos_loss_weight}")
    if lora_plus_ratio <= 0.0 or not math.isfinite(lora_plus_ratio):
        raise ValueError(f"Invalid lora_plus_ratio: {lora_plus_ratio}")
    if lower_layers_lr_mult < 0.0 or not math.isfinite(lower_layers_lr_mult):
        raise ValueError(f"Invalid lower_layers_lr_mult: {lower_layers_lr_mult}")

    if args.dry_run_check:
        summary = {
            "status": "dry_run_ok",
            "voice_name": cfg.get("voice_name"),
            "base_gguf_exists": base_gguf.is_file(),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "replay_train": len(replay_train),
            "replay_val": len(replay_val),
            "mode": mode,
            "lora_r": tr_cfg.get("lora_r", 16),
            "use_rslora": use_rslora,
            "lora_plus_ratio": lora_plus_ratio,
            "lower_layers_end": lower_layers_end,
            "lower_layers_lr_mult": lower_layers_lr_mult,
            "label_smoothing": label_smoothing,
            "eos_loss_weight": eos_loss_weight,
            "epochs": epochs,
            "save_every_epochs": save_every_epochs,
            "overlay_precision": overlay_precision,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.device == "cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        torch.cuda.reset_peak_memory_stats()
    elif args.allow_cpu_smoke:
        device = "cpu"
    else:
        raise RuntimeError(
            "GPU (CUDA/ROCm) required for training; pass --device cpu or --allow-cpu-smoke for isolated CPU runs."
        )

    # Pure FP32 precision discipline for exact gradient alignment against F32 base GGUF
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    neutral_instruct_prefixes: list[list[int]] = []
    neutral_instruct_texts = tr_cfg.get("neutral_instruct_pool", [])
    if instruct_prob > 0.0 and neutral_instruct_texts:
        instruction_prefix = ds_cfg.get("instruction_prefix", "You are a helpful assistant.")
        rt_dir = resolve_cfg_path(paths["cosyvoice_runtime_dir"], cfg_dir)
        with NativeCosyVoiceTokenizer(rt_dir, base_gguf) as tok:
            for inst_txt in neutral_instruct_texts:
                neutral_instruct_prefixes.append(tok.encode(f"{instruction_prefix} {inst_txt}<|endofprompt|>"))

    voice_name = cfg.get("voice_name", "custom_voice")
    run_dir = runs_root / voice_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite_run:
        raise FileExistsError(f"Run directory already exists and is non-empty: {run_dir} (pass --overwrite-run to overwrite)")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(json.dumps(split_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    t_load_start = time.perf_counter()
    print(f"[1/4] Loading base GGUF: {base_gguf}")
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"      Device: {device} ({dev_name}) | PyTorch: {torch.__version__} | Mode: {mode.upper()}")

    model = CosyVoice3LLM(
        gguf_path=base_gguf,
        device=device,
        mode=mode,
        lora_r=int(tr_cfg.get("lora_r", 16)),
        lora_alpha=float(tr_cfg.get("lora_alpha", 32.0)),
        lora_dropout=float(tr_cfg.get("lora_dropout", 0.1)),
        train_speech_head=train_speech_head,
        use_rslora=use_rslora,
        lower_layers_end=lower_layers_end,
        lower_layers_lr_mult=lower_layers_lr_mult,
        llm_decoder_lora_r=llm_decoder_lora_r,
    )

    if mode == "lora" and not train_speech_head:
        unexpected = [
            n for n, p in model.named_parameters()
            if p.requires_grad and not n.endswith(("lora_A", "lora_B"))
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable non-LoRA weights when train_speech_head=false: {unexpected}")

    param_groups = build_optimizer_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        lora_plus_ratio=lora_plus_ratio,
        lower_layers_end=lower_layers_end,
        lower_layers_lr_mult=lower_layers_lr_mult,
        head_lr_mult=head_lr_mult,
    )
    trainable_params = [p for g in param_groups for p in g["params"]]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"      Trainable params: {n_trainable:,} / {n_total:,} ({100.0 * n_trainable / n_total:.2f}%)")
    for g in param_groups:
        g_cnt = sum(p.numel() for p in g["params"])
        print(f"        - {g['name']:16s}: {g_cnt:10,d} params | lr={g['lr']:.2e} | wd={g['weight_decay']}")

    val_pairs = build_fixed_val_pairs(val_samples, train_samples, max_paired_tokens, seed=seed)
    init_eval = evaluate_suite(model, val_samples, val_pairs, prefix_tokens, replay_val)
    print(
        f"[2/4] Baseline eval | single_ce={init_eval['val_single_ce']} | "
        f"zeroshot_ce={init_eval['val_zeroshot_ce']} | replay_ce={init_eval['val_replay_ce']} | "
        f"eos_nll={init_eval['val_eos_loss']}"
    )
    if init_eval["val_single_ce"] is None or not math.isfinite(init_eval["val_single_ce"]):
        raise FloatingPointError("Non-finite initial validation loss")

    optimizer = torch.optim.AdamW(param_groups)
    replay_per_epoch = int(round(len(train_samples) * replay_ratio)) if (replay_train and replay_ratio > 0.0) else 0
    effective_epoch_size = len(train_samples) + replay_per_epoch
    steps_per_epoch = max(1, math.ceil(effective_epoch_size / grad_accum))
    total_steps = args.max_steps if args.max_steps is not None else (epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * float(tr_cfg.get("warmup_ratio", 0.1))))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_metric_val = init_eval.get(selection_metric)
    if best_metric_val is None:
        selection_metric = "val_single_ce"
        best_metric_val = init_eval["val_single_ce"]
    init_replay_ce = init_eval["val_replay_ce"]

    best_overlay_path = run_dir / "best_overlay.gguf"
    last_overlay_path = run_dir / "last_overlay.gguf"
    best_ckpt_path = run_dir / "best_llm_weights.pt"
    last_ckpt_path = run_dir / "last_llm_weights.pt"
    saved_overlays: list[dict] = []
    history: list[dict] = []

    print(
        f"[3/4] Training | train={len(train_samples)} (+replay {replay_per_epoch}/ep) | "
        f"val={len(val_samples)} (pairs={len(val_pairs)}, replay_val={len(replay_val)}) | "
        f"epochs={epochs} | total_steps={total_steps} | ls={label_smoothing} | eos_w={eos_loss_weight}"
    )

    rng = random.Random(seed)
    global_step = 0
    t_start = time.perf_counter()
    model.train()
    optimizer.zero_grad()

    for epoch in range(1, epochs + 1):
        epoch_items: list[tuple[dict, bool]] = [(s, False) for s in train_samples]
        if replay_per_epoch > 0:
            for _ in range(replay_per_epoch):
                epoch_items.append((rng.choice(replay_train), True))
        rng.shuffle(epoch_items)

        epoch_losses: list[float] = []
        accum_count = 0
        n_items = len(epoch_items)
        rem = n_items % grad_accum

        for idx, (sample, is_replay) in enumerate(epoch_items, 1):
            pair_pool = replay_train if is_replay else train_samples
            txt_ids, sp_ids, loss_start = build_training_example(
                sample,
                pair_pool,
                prefix_tokens,
                pair_prob if not is_replay else 0.5,
                max_paired_tokens,
                rng,
                instruct_prob=instruct_prob,
                neutral_instruct_prefixes=neutral_instruct_prefixes,
                is_replay=is_replay,
            )
            loss = model.forward_loss(
                txt_ids,
                sp_ids,
                loss_start_speech_idx=loss_start,
                label_smoothing=label_smoothing,
                eos_loss_weight=eos_loss_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, step={global_step}, sample={sample['id']}")
            epoch_losses.append(float(loss.item()))

            divisor = rem if (rem and idx > n_items - rem) else grad_accum
            (loss / divisor).backward()
            accum_count += 1

            if accum_count == grad_accum or idx == n_items:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if args.max_steps is not None and global_step >= args.max_steps:
                    break

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        ev = evaluate_suite(model, val_samples, val_pairs, prefix_tokens, replay_val)
        cur_metric = ev[selection_metric]
        if cur_metric is None or not math.isfinite(cur_metric):
            raise FloatingPointError(f"Non-finite validation metric {selection_metric} at epoch={epoch}")

        replay_drift = None
        drift_flag = ""
        if init_replay_ce is not None and ev["val_replay_ce"] is not None:
            replay_drift = round(ev["val_replay_ce"] - init_replay_ce, 5)
            if replay_drift > replay_drift_warn:
                drift_flag = f" [WARN replay_drift=+{replay_drift:.4f}]"

        cur_lr = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t_start
        is_best = cur_metric <= best_metric_val
        if is_best:
            best_metric_val = cur_metric
            if mode == "lora":
                model.export_voice_overlay_gguf(
                    best_overlay_path,
                    precision=overlay_precision,
                    extra_metadata={"cosyvoice.overlay.voice_name": voice_name, "cosyvoice.overlay.epoch": epoch},
                )
            if save_merged_pt or mode == "full":
                torch.save(model.export_gguf_tensor_dict(torch.float32), best_ckpt_path)

        if mode == "lora" and (epoch % save_every_epochs == 0 or epoch == epochs):
            ep_path = ckpt_dir / f"epoch_{epoch:02d}.gguf"
            model.export_voice_overlay_gguf(
                ep_path,
                precision=overlay_precision,
                extra_metadata={"cosyvoice.overlay.voice_name": voice_name, "cosyvoice.overlay.epoch": epoch},
            )
            saved_overlays.append({
                "epoch": epoch,
                "step": global_step,
                "path": str(ep_path),
                "bytes": ep_path.stat().st_size,
                **ev,
                "replay_drift": replay_drift,
            })

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | step {global_step:04d} | tr={train_loss:.4f} | "
            f"val_s={ev['val_single_ce']:.4f} | val_zs={ev['val_zeroshot_ce']:.4f} | "
            f"val_rep={ev['val_replay_ce'] if ev['val_replay_ce'] is not None else '-'} | "
            f"eos={ev['val_eos_loss']:.4f} {'(*BEST*)' if is_best else ''}{drift_flag} | "
            f"lr={cur_lr:.2e} | {elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "step": global_step,
            "train_loss": round(train_loss, 5),
            **ev,
            "replay_drift": replay_drift,
            "lr": cur_lr,
            "elapsed_sec": round(elapsed, 2),
        })

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if mode == "lora":
        model.export_voice_overlay_gguf(
            last_overlay_path,
            precision=overlay_precision,
            extra_metadata={"cosyvoice.overlay.voice_name": voice_name, "cosyvoice.overlay.epoch": epochs},
        )
        if not best_overlay_path.is_file():
            model.export_voice_overlay_gguf(best_overlay_path, precision=overlay_precision)
    if save_merged_pt or mode == "full":
        torch.save(model.export_gguf_tensor_dict(torch.float32), last_ckpt_path)
        if not best_ckpt_path.is_file():
            torch.save(model.export_gguf_tensor_dict(torch.float32), best_ckpt_path)

    peak_alloc = 0.0
    peak_res = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() / 1048576.0
        peak_res = torch.cuda.max_memory_reserved() / 1048576.0

    metrics = {
        "schema": 2,
        "voice_name": voice_name,
        "mode": mode,
        "parameter_dtype": "float32",
        "overlay_precision": overlay_precision if mode == "lora" else None,
        "selection_metric": selection_metric,
        "initial_eval": init_eval,
        "best_selection_value": round(float(best_metric_val), 5),
        "total_steps": global_step,
        "gpu_peak_allocated_mib": round(peak_alloc, 2),
        "gpu_peak_reserved_mib": round(peak_res, 2),
        "wall_seconds": round(time.perf_counter() - t_load_start, 2),
        "best_overlay": str(best_overlay_path) if best_overlay_path.is_file() else None,
        "last_overlay": str(last_overlay_path) if last_overlay_path.is_file() else None,
        "saved_overlays": saved_overlays,
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path.is_file() else None,
        "last_checkpoint": str(last_ckpt_path) if last_ckpt_path.is_file() else None,
        "history": history,
    }
    metrics_path = run_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] Training complete! Best output: {best_overlay_path if best_overlay_path.is_file() else best_ckpt_path}")


if __name__ == "__main__":
    main()
