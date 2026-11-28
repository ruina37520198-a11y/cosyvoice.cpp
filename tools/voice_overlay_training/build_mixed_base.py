"""Build a quality-oriented mixed-precision CosyVoice3-2512 shared base GGUF.

Keeps LLM transformer layers (`layers.*`), Flow/DiT (`decoder.*`), and CPU-mapped
`embed_tokens.*` in F16 so native Voice Overlays (`cosyvoice3-voice-overlay-v1`) and
runtime LoRA adapters remain bit-compatible, while preserving the HiFT vocoder, F0
predictor, `llm_decoder`, `speech_embedding`, and all 1D norms/biases in native F32
from the official F32 GGUF (+~65 MB over standard F16).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import gguf
import numpy as np


BASE_ARCHITECTURE = "cosyvoice3-2512"
KEEP_F16_PREFIXES = ("layers.", "decoder.", "embed_tokens.")
FLOATS = {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_architecture(reader: gguf.GGUFReader, path: Path) -> str:
    field = reader.get_field("general.architecture")
    if field is None:
        raise ValueError(f"Missing general.architecture in {path.name}")
    arch = bytes(field.parts[field.data[0]]).decode("utf-8")
    if arch != BASE_ARCHITECTURE:
        raise ValueError(f"Expected architecture {BASE_ARCHITECTURE} in {path.name}, got {arch}")
    return arch


def _copy_field(writer: gguf.GGUFWriter, key: str, field) -> None:
    if key.startswith("GGUF.") or key == "general.architecture":
        return
    types = field.types
    kind = types[0]
    if kind == gguf.GGUFValueType.ARRAY:
        sub = types[1]
        if sub == gguf.GGUFValueType.STRING:
            values = [bytes(field.parts[idx]).decode("utf-8") for idx in field.data]
        else:
            values = [field.parts[idx].tolist()[0] for idx in field.data]
        writer.add_array(key, values)
    elif kind == gguf.GGUFValueType.STRING:
        writer.add_string(key, bytes(field.parts[field.data[0]]).decode("utf-8"))
    elif kind == gguf.GGUFValueType.UINT32:
        writer.add_uint32(key, int(field.parts[field.data[0]][0]))
    elif kind == gguf.GGUFValueType.INT32:
        writer.add_int32(key, int(field.parts[field.data[0]][0]))
    elif kind == gguf.GGUFValueType.FLOAT32:
        writer.add_float32(key, float(field.parts[field.data[0]][0]))
    elif kind == gguf.GGUFValueType.BOOL:
        writer.add_bool(key, bool(field.parts[field.data[0]][0]))
    else:
        raise ValueError(f"Unsupported GGUF metadata type for {key}: {types}")


def _should_keep_f16(name: str, tensor, ref_f16_tensor=None) -> bool:
    if tensor.data.ndim < 2 or not name.startswith(KEEP_F16_PREFIXES):
        return False
    if ref_f16_tensor is not None:
        return ref_f16_tensor.tensor_type == gguf.GGMLQuantizationType.F16
    return True


def _close_gguf_reader(reader: gguf.GGUFReader | None) -> None:
    if reader is None:
        return
    mmap_obj = getattr(getattr(reader, "data", None), "_mmap", None)
    if mmap_obj is not None:
        try:
            mmap_obj.close()
        except Exception:
            pass


def _divergent_layer(path: Path, reference: dict[str, object]) -> str | None:
    reader = gguf.GGUFReader(path)
    try:
        for tensor in reader.tensors:
            if tensor.name.startswith("layers."):
                ref = reference[tensor.name]
                if (
                    tensor.tensor_type != ref.tensor_type
                    or not np.array_equal(tensor.data.view(np.uint8), ref.data.view(np.uint8))
                ):
                    return tensor.name
        return None
    finally:
        _close_gguf_reader(reader)


def build_mixed_base(f32_path: Path | str, output: Path | str, *, f16_path: Path | str | None = None) -> dict:
    f32_path = Path(f32_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)

    r32 = gguf.GGUFReader(f32_path)
    r16 = None
    try:
        arch = _read_architecture(r32, f32_path)
        t32_map = {tensor.name: tensor for tensor in r32.tensors}

        t16_map: dict[str, object] | None = None
        if f16_path is not None:
            f16_path = Path(f16_path)
            r16 = gguf.GGUFReader(f16_path)
            _read_architecture(r16, f16_path)
            t16_map = {tensor.name: tensor for tensor in r16.tensors}
            if t16_map.keys() != t32_map.keys():
                raise ValueError("F16 reference and F32 base tensor inventories differ")

        meta_source = r16 if r16 is not None else r32
        f16_count = 0
        f32_upgraded = 0
        f32_already = 0

        with tempfile.TemporaryDirectory(prefix=".mixed-base-", dir=output.parent) as tmp:
            temporary = Path(tmp) / "mixed_base.gguf"
            writer = gguf.GGUFWriter(temporary, arch)
            try:
                saw_file_type = False
                for key, field in meta_source.fields.items():
                    if key == "general.file_type":
                        writer.add_uint32(key, 1)
                        saw_file_type = True
                        continue
                    if key == "general.quantized_by":
                        writer.add_string(key, "cosyvoice-mixed-hift-f32")
                        continue
                    _copy_field(writer, key, field)
                if not saw_file_type:
                    writer.add_uint32("general.file_type", 1)

                ordered = r16.tensors if r16 is not None else r32.tensors
                for entry in ordered:
                    name = entry.name
                    t32 = t32_map[name]
                    t16 = t16_map[name] if t16_map is not None else None
                    if t32.tensor_type not in FLOATS:
                        raise ValueError(f"Unsupported non-floating tensor in F32 base: {name}")
                    if t16 is not None and (t16.tensor_type not in FLOATS or not np.array_equal(t16.shape, t32.shape)):
                        raise ValueError(f"Incompatible tensor in F16 reference: {name}")

                    if _should_keep_f16(name, t32, t16):
                        if t16 is not None:
                            arr = np.ascontiguousarray(t16.data, dtype=np.float16)
                        else:
                            arr = np.ascontiguousarray(t32.data.astype(np.float16))
                        if not np.isfinite(arr).all():
                            raise ValueError(f"Non-finite F16 values in {name}")
                        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
                        f16_count += 1
                    else:
                        if t32.tensor_type != gguf.GGMLQuantizationType.F32:
                            raise ValueError(f"Expected native F32 tensor in F32 base: {name}")
                        arr = np.ascontiguousarray(t32.data, dtype=np.float32)
                        if not np.isfinite(arr).all():
                            raise ValueError(f"Non-finite F32 values in {name}")
                        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
                        if t16 is not None and t16.tensor_type == gguf.GGMLQuantizationType.F16:
                            f32_upgraded += 1
                        elif t16 is None and t32.data.ndim >= 2:
                            f32_upgraded += 1
                        else:
                            f32_already += 1

                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file(progress=False)
            finally:
                writer.close()

            layers_bit_identical = None
            if t16_map is not None:
                mismatch = _divergent_layer(temporary, t16_map)
                if mismatch is not None:
                    raise ValueError(f"Layer tensor diverged from F16 reference: {mismatch}")
                layers_bit_identical = True

            size = temporary.stat().st_size
            checksum = _sha256(temporary)
            if os.name == "nt":
                os.rename(temporary, output)
            else:
                os.link(temporary, output)
    finally:
        _close_gguf_reader(r16)
        _close_gguf_reader(r32)

    return {
        "bytes": size,
        "sha256": checksum,
        "f16_tensors": f16_count,
        "f32_upgraded_tensors": f32_upgraded,
        "f32_kept_tensors": f32_already,
        "layers_bit_identical_to_f16": layers_bit_identical,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f32", type=Path, required=True, help="Official CosyVoice3-2512 F32 base GGUF")
    parser.add_argument("--f16", type=Path, default=None, help="Optional official F16 GGUF for bit-identical layers.* copy")
    parser.add_argument("--out", type=Path, required=True, help="Output path for mixed-precision base GGUF")
    args = parser.parse_args()
    print(json.dumps(build_mixed_base(args.f32, args.out, f16_path=args.f16), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
