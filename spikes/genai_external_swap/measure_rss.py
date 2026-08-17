"""RSS sampler for the GenAI external-data-swap spike (#10, Gate 0.1 step 7).

Snapshots resident set size (VmRSS) around load / first-token so mmap-vs-copy can be judged: with a
file-path load ORT *can* mmap external initializers, so RSS-after-load close to the external file size
indicates mmap; ~2x indicates a copy. Uses /proc/self/status on Linux (also works on Android), falling
back to psutil if present.
"""

from __future__ import annotations

from pathlib import Path


def rss_kb() -> int:
    """Current process resident set size in kB (-1 if unavailable)."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process().memory_info().rss // 1024
    except Exception:  # pragma: no cover - best-effort sampler
        return -1


class RssTrace:
    """Named RSS checkpoints, printed as a small table with deltas."""

    def __init__(self) -> None:
        self.points: list[tuple[str, int]] = []

    def mark(self, label: str) -> int:
        v = rss_kb()
        self.points.append((label, v))
        return v

    def report(self) -> str:
        lines = ["RSS (kB):"]
        base = self.points[0][1] if self.points else 0
        for label, v in self.points:
            lines.append(f"  {label:<16} {v:>10}  (+{v - base})")
        return "\n".join(lines)


__all__ = ["rss_kb", "RssTrace"]
