"""#12 desktop correctness invariant (Gate 0.2): loading the File #9 inference graph with the mmap /
external-initializers config keys must produce **byte-identical** first-token logits to the default
(copied-buffer) load. Any divergence is a hard fail — mmap must be transparent, only cheaper.

This is the env-gated desktop leg (needs `onnxruntime`; run under the export or genai-smoke profile). The
four-point RSS win itself is measured on-device (the manual Gate 0.2 table); here we only prove
correctness + report desktop RSS deltas around each load.

Run:  uv run --python 3.12 --group genai-smoke python -m spikes.mmap.base_blob_mmap_spike \
          --dir build/pkg/variants/cpu-int4/inference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from spikes.mmap.measure_rss import RssTrace

# ORT config keys under test (external-initializers folder + the ORT-format/bytes toggles).
EXTERNAL_FOLDER_KEY = "session.model_external_initializers_file_folder_path"
USE_BYTES_KEY = "session.use_ort_model_bytes_for_initializers"


def _build_dummy_inputs(sess, genai_config: dict) -> dict:
    """Minimal single-token decoder inputs (empty KV cache) from the genai_config geometry."""
    decoder = genai_config.get("model", {}).get("decoder", {})
    n_layers = int(decoder.get("num_hidden_layers", 0))
    n_kv = int(decoder.get("num_key_value_heads", decoder.get("num_attention_heads", 0)))
    head = int(decoder.get("head_size", 0))
    feed: dict[str, np.ndarray] = {}
    names = {i.name for i in sess.get_inputs()}
    if "input_ids" in names:
        feed["input_ids"] = np.array([[1]], dtype=np.int64)
    if "attention_mask" in names:
        feed["attention_mask"] = np.array([[1]], dtype=np.int64)
    if "position_ids" in names:
        feed["position_ids"] = np.array([[0]], dtype=np.int64)
    for i in range(n_layers):
        for kind in ("key", "value"):
            name = f"past_key_values.{i}.{kind}"
            if name in names:
                feed[name] = np.zeros((1, n_kv, 0, head), dtype=np.float32)
    return feed


def _first_logits(inf_dir: Path, use_mmap_keys: bool, trace: RssTrace, tag: str) -> np.ndarray:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if use_mmap_keys:
        opts.add_session_config_entry(EXTERNAL_FOLDER_KEY, str(inf_dir))
        opts.add_session_config_entry(USE_BYTES_KEY, "0")
    trace.mark(f"{tag}:pre")
    sess = ort.InferenceSession(str(inf_dir / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    trace.mark(f"{tag}:loaded")
    genai_config = json.loads((inf_dir / "genai_config.json").read_text()) if (inf_dir / "genai_config.json").is_file() else {}
    feed = _build_dummy_inputs(sess, genai_config)
    out = sess.run(None, feed)
    trace.mark(f"{tag}:ran")
    return np.asarray(out[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="mmap external-initializer correctness invariant (#12)")
    ap.add_argument("--dir", required=True, help="File #9 inference dir (model.onnx + weight_handoff_map.json)")
    ap.add_argument("--rtol", type=float, default=0.0, help="allowed rtol (default 0 = byte-identical)")
    args = ap.parse_args()

    inf_dir = Path(args.dir)
    if not (inf_dir / "model.onnx").is_file():
        print(f"FAIL: no model.onnx in {inf_dir}")
        return 2

    trace = RssTrace()
    logits_copy = _first_logits(inf_dir, use_mmap_keys=False, trace=trace, tag="copy")
    logits_mmap = _first_logits(inf_dir, use_mmap_keys=True, trace=trace, tag="mmap")

    print(trace.report())
    identical = np.array_equal(logits_copy, logits_mmap) or np.allclose(
        logits_copy, logits_mmap, rtol=args.rtol, atol=0.0
    )
    max_diff = float(np.nanmax(np.abs(logits_copy - logits_mmap))) if logits_copy.size else 0.0
    print(f"|logits_copy - logits_mmap| max = {max_diff:.6g}")
    if identical:
        print("PASS: external-initializer config load is byte-identical to the copy baseline.")
        return 0
    print("FAIL: logits differ — mmap/external-initializer path is not transparent (Gate 0.2 hard fail).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
