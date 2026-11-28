"""Cross-platform GGUF v3 utilities and native cosyvoice.cpp tokenizer bridge.

Provides:
  - Minimal zero-dependency GGUF v3 parser and tensor loader (F32/F16/I8/I32)
  - Atomic writer for `cosyvoice3-voice-overlay-v1` LoRA / override GGUF files
  - Cross-platform binary/shared-library discovery for `cosyvoice-cli` and `libcosyvoice`
  - Optional Traditional -> Simplified Chinese normalization (preserving Japanese Kana/Kanji)
"""
from __future__ import annotations

import ctypes
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

_KANA_RE = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

OVERLAY_ARCHITECTURE = "cosyvoice3-voice-overlay"
OVERLAY_FORMAT = "cosyvoice3-voice-overlay-v1"
BASE_ARCHITECTURE = "cosyvoice3-2512"


def _bind_lcmap_string_ex():
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    mapper = kernel32.LCMapStringEx
    mapper.argtypes = (
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_wchar_p, ctypes.c_int,
        ctypes.c_wchar_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t,
    )
    mapper.restype = ctypes.c_int
    return mapper


_LCMAP_STRING_EX = _bind_lcmap_string_ex()
_OPENCC_CONVERTER = None
_OPENCC_CHECKED = False


def _get_opencc_converter():
    global _OPENCC_CONVERTER, _OPENCC_CHECKED
    if not _OPENCC_CHECKED:
        _OPENCC_CHECKED = True
        try:
            import opencc  # type: ignore

            _OPENCC_CONVERTER = opencc.OpenCC("t2s")
        except Exception:
            _OPENCC_CONVERTER = None
    return _OPENCC_CONVERTER


def normalize_transcript_text(text: str, simplify_chinese: bool = True) -> str:
    """Normalize whitespace and convert Traditional Chinese to Simplified when not Japanese."""
    cleaned = " ".join(text.strip().split())
    if not cleaned or not simplify_chinese:
        return cleaned
    # Preserve Japanese transcripts intact whenever hiragana or katakana is present.
    if _KANA_RE.search(cleaned) or not _CJK_RE.search(cleaned):
        return cleaned
    if os.name == "nt":
        mapper = _LCMAP_STRING_EX or _bind_lcmap_string_ex()
        if mapper is not None:
            flags = 0x02000000  # LCMAP_SIMPLIFIED_CHINESE
            length = mapper("zh-CN", flags, cleaned, -1, None, 0, None, None, 0)
            if length > 0:
                output = ctypes.create_unicode_buffer(length)
                if mapper("zh-CN", flags, cleaned, -1, output, length, None, None, 0) != 0:
                    return output.value
    converter = _get_opencc_converter()
    if converter is not None:
        return converter.convert(cleaned)
    return cleaned


def find_cosyvoice_cli(runtime_dir: str | Path | None = None) -> Path:
    """Locate `cosyvoice-cli` (or `cosyvoice-cli.exe` on Windows) in runtime_dir or PATH."""
    names = ["cosyvoice-cli.exe", "cosyvoice-cli"] if os.name == "nt" else ["cosyvoice-cli"]
    if runtime_dir is not None:
        rt = Path(runtime_dir).resolve()
        for sub in (rt, rt / "bin"):
            for name in names:
                cand = sub / name
                if cand.is_file():
                    return cand
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    raise FileNotFoundError(
        f"Could not find cosyvoice-cli in {runtime_dir or 'PATH'}. "
        "Pass --runtime-dir pointing to your cosyvoice.cpp build/release directory."
    )


def find_cosyvoice_library(runtime_dir: str | Path) -> Path:
    """Locate the native cosyvoice shared library (cosyvoice.dll / libcosyvoice.so / libcosyvoice.dylib)."""
    rt = Path(runtime_dir).resolve()
    search_dirs = [d for d in (rt, rt / "lib", rt / "bin", rt.parent / "lib") if d.is_dir()]
    if os.name == "nt":
        candidates = ["cosyvoice.dll", "libcosyvoice.dll"]
    elif sys.platform == "darwin":
        candidates = ["libcosyvoice.dylib"]
    else:
        candidates = ["libcosyvoice.so"]

    for d in search_dirs:
        for name in candidates:
            p = d / name
            if p.is_file():
                return p
        if os.name != "nt":
            pattern = "libcosyvoice*.dylib" if sys.platform == "darwin" else "libcosyvoice.so*"
            matches = sorted(d.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Could not locate cosyvoice shared library under {rt}. "
        "Ensure cosyvoice.dll / libcosyvoice.so / libcosyvoice.dylib is present in runtime_dir."
    )


def find_ffmpeg_binary(tool: str, runtime_dir: Path | None = None) -> Path | None:
    """Locate `ffmpeg` or `ffprobe` across Windows, Linux, and macOS."""
    base = tool[:-4] if tool.lower().endswith(".exe") else tool
    names = [f"{base}.exe", base] if os.name == "nt" else [base]
    if runtime_dir is not None:
        for sub in (runtime_dir, runtime_dir / "bin"):
            for name in names:
                cand = sub / name
                if cand.is_file():
                    return cand
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _read_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _read_val(f, vtype: int):
    if vtype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:
        return struct.unpack("<?", f.read(1))[0]
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        atype, alen = struct.unpack("<IQ", f.read(12))
        return [_read_val(f, atype) for _ in range(alen)]
    if vtype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def parse_gguf_metadata_and_tensors(path: str | Path):
    """Return `(metadata_dict, tensors_dict, data_offset)`."""
    with open(path, "rb") as f:
        magic, _version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF file: {path}")
        metadata = {}
        for _ in range(n_kv):
            key = _read_str(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            metadata[key] = _read_val(f, vtype)
        alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        if alignment <= 0:
            alignment = GGUF_DEFAULT_ALIGNMENT
        tensors = {}
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors[name] = (dims, ttype, offset)
        pos = f.tell()
        data_offset = ((pos + alignment - 1) // alignment) * alignment
    return metadata, tensors, data_offset


def parse_gguf_tensors(path: str | Path):
    """Return `(tensors_dict, data_offset)` where `tensors_dict[name] = (dims, ttype, rel_offset)`."""
    _meta, tensors, data_offset = parse_gguf_metadata_and_tensors(path)
    return tensors, data_offset


def load_gguf_numpy(buf, data_offset: int, tensor_info) -> np.ndarray:
    """Load a GGUF tensor (F32=0, F16=1, I8=24, I32=26) into a row-major NumPy array."""
    dims, ttype, offset = tensor_info
    nelems = 1
    for d in dims:
        nelems *= d
    byte_off = data_offset + offset
    if ttype == 0:
        arr = np.frombuffer(buf, dtype=np.float32, count=nelems, offset=byte_off)
    elif ttype == 1:
        arr = np.frombuffer(buf, dtype=np.float16, count=nelems, offset=byte_off)
    elif ttype == 24:
        arr = np.frombuffer(buf, dtype=np.int8, count=nelems, offset=byte_off)
    elif ttype == 26:
        arr = np.frombuffer(buf, dtype=np.int32, count=nelems, offset=byte_off)
    else:
        raise ValueError(f"Unsupported GGUF tensor type {ttype}")
    return arr.copy().reshape(dims[::-1])


def _pack_gguf_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_voice_overlay_gguf(
    out_path: str | Path,
    tensors: dict[str, np.ndarray],
    *,
    tensor_scales: dict[str, float] | None = None,
    default_scale: float = 1.0,
    extra_metadata: dict[str, str | float | int] | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a native `cosyvoice3-voice-overlay-v1` GGUF v3 file atomically.

    Tensor arrays must be C-contiguous float32 or float16 NumPy arrays in PyTorch/NumPy row-major
    shape `(out, in)` (or `(rank, in)` for `.lora_a`, `(out, rank)` for `.lora_b`). Dimensions are
    serialized in GGML column-major order (`shape[::-1]`) to match `cosyvoice-voice-overlay.cpp`.
    """
    out = Path(out_path).resolve()
    if out.exists() and not overwrite:
        raise FileExistsError(f"Overlay file already exists: {out}")
    if not tensors:
        raise ValueError("Cannot write an empty voice overlay GGUF")
    if not math.isfinite(default_scale) or default_scale <= 0.0:
        raise ValueError(f"default_scale must be finite and positive, got {default_scale}")

    out.parent.mkdir(parents=True, exist_ok=True)

    kv_entries: list[tuple[str, int, bytes]] = [
        ("general.architecture", GGUF_TYPE_STRING, _pack_gguf_str(OVERLAY_ARCHITECTURE)),
        ("general.alignment", GGUF_TYPE_UINT32, struct.pack("<I", GGUF_DEFAULT_ALIGNMENT)),
        ("cosyvoice.overlay.format", GGUF_TYPE_STRING, _pack_gguf_str(OVERLAY_FORMAT)),
        ("cosyvoice.overlay.base_architecture", GGUF_TYPE_STRING, _pack_gguf_str(BASE_ARCHITECTURE)),
        ("cosyvoice.overlay.default_scale", GGUF_TYPE_FLOAT32, struct.pack("<f", float(default_scale))),
    ]
    if tensor_scales:
        for base_name in sorted(tensor_scales):
            sc = float(tensor_scales[base_name])
            if not math.isfinite(sc) or sc <= 0.0:
                raise ValueError(f"Invalid tensor scale for {base_name}: {sc}")
            kv_entries.append(
                (f"cosyvoice.overlay.tensor_scale.{base_name}", GGUF_TYPE_FLOAT32, struct.pack("<f", sc))
            )
    if extra_metadata:
        for key in sorted(extra_metadata):
            val = extra_metadata[key]
            if isinstance(val, str):
                kv_entries.append((key, GGUF_TYPE_STRING, _pack_gguf_str(val)))
            elif isinstance(val, bool):
                kv_entries.append((key, GGUF_TYPE_UINT32, struct.pack("<I", int(val))))
            elif isinstance(val, int):
                kv_entries.append((key, GGUF_TYPE_UINT32, struct.pack("<I", val)))
            elif isinstance(val, float):
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite metadata float for {key}")
                kv_entries.append((key, GGUF_TYPE_FLOAT32, struct.pack("<f", val)))
            else:
                raise TypeError(f"Unsupported extra_metadata type for {key}: {type(val).__name__}")

    prepared: list[tuple[str, np.ndarray, int, int]] = []
    cur_offset = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        if arr.dtype == np.float32:
            ttype = GGML_TYPE_F32
        elif arr.dtype == np.float16:
            ttype = GGML_TYPE_F16
        else:
            raise ValueError(f"Overlay tensor '{name}' must be float16 or float32, got {arr.dtype}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Overlay tensor '{name}' contains non-finite values")
        prepared.append((name, arr, ttype, cur_offset))
        nbytes = arr.nbytes
        cur_offset += ((nbytes + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT

    tmp = out.with_name(f".{out.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<4sIQQ", GGUF_MAGIC, GGUF_VERSION, len(prepared), len(kv_entries)))
            for key, vtype, packed_val in kv_entries:
                f.write(_pack_gguf_str(key))
                f.write(struct.pack("<I", vtype))
                f.write(packed_val)
            for name, arr, ttype, rel_off in prepared:
                f.write(_pack_gguf_str(name))
                dims_ggml = arr.shape[::-1]
                f.write(struct.pack("<I", len(dims_ggml)))
                for d in dims_ggml:
                    f.write(struct.pack("<Q", int(d)))
                f.write(struct.pack("<IQ", ttype, rel_off))
            pos = f.tell()
            pad = ((pos + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT - pos
            if pad:
                f.write(b"\x00" * pad)
            for _name, arr, _ttype, _rel_off in prepared:
                raw = arr.tobytes()
                f.write(raw)
                tpad = ((len(raw) + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT - len(raw)
                if tpad:
                    f.write(b"\x00" * tpad)
        os.replace(tmp, out)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return out


class NativeCosyVoiceTokenizer:
    """Wrapper around `cosyvoice.cpp`'s native BPE tokenizer loaded directly from a GGUF model."""

    def __init__(self, runtime_dir: str | Path, gguf_path: str | Path):
        lib_path = find_cosyvoice_library(runtime_dir)
        lib_dir = lib_path.parent
        self._dll_dir_handle = None
        if os.name == "nt":
            if hasattr(os, "add_dll_directory"):
                self._dll_dir_handle = os.add_dll_directory(str(lib_dir))
            os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")

        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.cosyvoice_tokenizer_load_from_file.argtypes = [ctypes.c_char_p]
        self.lib.cosyvoice_tokenizer_load_from_file.restype = ctypes.c_void_p
        self.lib.cosyvoice_tokenizer_free.argtypes = [ctypes.c_void_p]
        self.lib.cosyvoice_tokenization_result_create.argtypes = []
        self.lib.cosyvoice_tokenization_result_create.restype = ctypes.c_void_p
        self.lib.cosyvoice_tokenization_result_free.argtypes = [ctypes.c_void_p]
        self.lib.cosyvoice_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        self.lib.cosyvoice_tokenize.restype = ctypes.c_int
        self.lib.cosyvoice_tokenization_result_get_n_tokens.argtypes = [ctypes.c_void_p]
        self.lib.cosyvoice_tokenization_result_get_n_tokens.restype = ctypes.c_size_t
        self.lib.cosyvoice_tokenization_result_get_tokens.argtypes = [ctypes.c_void_p]
        self.lib.cosyvoice_tokenization_result_get_tokens.restype = ctypes.POINTER(ctypes.c_int32)

        self._tok = self.lib.cosyvoice_tokenizer_load_from_file(str(Path(gguf_path).resolve()).encode("utf-8"))
        if not self._tok:
            raise RuntimeError(f"Failed to load native tokenizer from {gguf_path}")

    def encode(self, text: str) -> list[int]:
        res = self.lib.cosyvoice_tokenization_result_create()
        try:
            rc = self.lib.cosyvoice_tokenize(self._tok, text.encode("utf-8"), res)
            if rc < 0:
                raise RuntimeError(f"cosyvoice_tokenize failed with code {rc}")
            n = self.lib.cosyvoice_tokenization_result_get_n_tokens(res)
            ptr = self.lib.cosyvoice_tokenization_result_get_tokens(res)
            return [int(ptr[i]) for i in range(n)]
        finally:
            self.lib.cosyvoice_tokenization_result_free(res)

    def close(self):
        if getattr(self, "_tok", None):
            self.lib.cosyvoice_tokenizer_free(self._tok)
            self._tok = None
        if getattr(self, "_dll_dir_handle", None) is not None:
            self._dll_dir_handle.close()
            self._dll_dir_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
