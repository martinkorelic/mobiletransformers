"""Desktop GenAI external-data-swap spike (#10, Gate 0.1 steps 2-7).

Proves finding F2 on the desktop before the device port: overwriting the external weight bytes changes
GenAI's generated logits, and a **fresh** ``OgaCreateModel`` picks up the new bytes (no graph rewrite, no
fork). Works on either layout:

- a File #9 per-tensor package (``weight_handoff_map.json`` + per-tensor ``<name>.bin``): perturbs exactly
  one trainable ``.bin`` (never ``frozen_base.onnx.data``), refreshing its ``.sha256``;
- a builder-produced single-blob model (``model.onnx.data``): perturbs a byte range in the blob.

Run under the genai profile (Python >=3.11):
    uv run --python 3.12 --group genai-smoke python spikes/genai_external_swap/desktop_spike.py --dir <inference_dir>

Exit 0 = swap observed (logits differ); non-zero = no effect (folded/copied) or load failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from spikes.genai_external_swap.measure_rss import RssTrace


def _first_logits(model_dir: str, prompt: str, trace: RssTrace | None = None) -> np.ndarray:
    """One greedy token from a FRESH model (GenAI caches externals at construction — no reuse)."""
    import onnxruntime_genai as og  # noqa: PLC0415

    if trace:
        trace.mark("pre-load")
    model = og.Model(model_dir)
    if trace:
        trace.mark("post-Model")
    tok = og.Tokenizer(model)
    params = og.GeneratorParams(model)
    ids = tok.encode(prompt)
    params.set_search_options(do_sample=False, max_length=len(ids) + 1)
    gen = og.Generator(model, params)
    gen.append_tokens(ids)
    gen.generate_next_token()
    logits = np.array(gen.get_output("logits"))[0, -1, :]
    if trace:
        trace.mark("post-first-token")
    return logits


def _perturb_target(inf_dir: Path) -> Path:
    """Choose the external file to perturb and return its path (backing it up first)."""
    handoff = inf_dir / "weight_handoff_map.json"
    if handoff.is_file():
        hmap = json.loads(handoff.read_text())
        loc = hmap["entries"][0]["externalDataLocation"]
        rel = loc.get("weight") or next(iter(loc.values()))
        return inf_dir / rel
    # builder single-blob layout: perturb the largest *.data / *.bin that is not the frozen base blob.
    candidates = [
        p
        for p in inf_dir.iterdir()
        if p.suffix in (".data", ".bin") and p.name != "frozen_base.onnx.data"
    ]
    if not candidates:
        raise SystemExit(f"no external weight file to perturb in {inf_dir}")
    return max(candidates, key=lambda p: p.stat().st_size)


def _apply_delta(path: Path) -> None:
    """Simulate a merge delta: scale a wide contiguous float32 region by 1.5 so on-path weights change
    measurably (a 64-byte low-mantissa XOR is too weak — it can land entirely in unused embedding rows).
    Deterministic; NaN/Inf clamped. Refreshes a sibling .sha256 if present."""
    buf = bytearray(path.read_bytes())
    n = len(buf)
    start = (n // 10) * 3  # 30% in — past most of the embedding table, into transformer weights
    span = min(8 * 1024 * 1024, n - start)
    span -= span % 4
    if span > 0:
        region = np.frombuffer(bytes(buf[start : start + span]), dtype="<f4").copy()
        region = np.nan_to_num(region * np.float32(1.5), posinf=1e4, neginf=-1e4).astype("<f4")
        buf[start : start + span] = region.tobytes()
    path.write_bytes(bytes(buf))
    sha = path.with_suffix(path.suffix + ".sha256")
    if sha.exists():
        sha.write_text(hashlib.sha256(bytes(buf)).hexdigest() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="GenAI external-data-swap desktop spike (Gate 0.1)")
    ap.add_argument("--dir", required=True, help="GenAI-loadable inference dir (model.onnx + genai_config.json)")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    inf_dir = Path(args.dir)
    if not (inf_dir / "genai_config.json").is_file():
        print(f"FAIL: no genai_config.json in {inf_dir} (not a GenAI package)")
        return 2

    trace = RssTrace()
    l_base = _first_logits(str(inf_dir), args.prompt, trace)

    target = _perturb_target(inf_dir)
    backup = target.with_suffix(target.suffix + ".spikebak")
    shutil.copy2(target, backup)
    print(f"perturbing external weight: {target.name} ({target.stat().st_size} bytes)")
    try:
        _apply_delta(target)
        l_swap = _first_logits(str(inf_dir), args.prompt)
    finally:
        shutil.move(str(backup), str(target))  # restore original bytes

    differ = not np.allclose(l_base, l_swap, rtol=args.rtol, equal_nan=False)
    print(trace.report())
    print(f"|L_base - L_swap| max = {np.nanmax(np.abs(l_base - l_swap)):.6g}")
    if differ:
        print("PASS: external swap observed on fresh OgaCreateModel (logits differ).")
        return 0
    print("FAIL: logits identical after swap — trainable externals folded or copied (Gate 0.1 hard fail).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
