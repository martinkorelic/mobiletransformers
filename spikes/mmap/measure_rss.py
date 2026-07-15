"""#12: RSS sampler for the mmap experiments — a thin re-export of the genai spike's sampler.

The plan is explicit: do NOT write a second sampler. Import `RssTrace`/`rss_kb` from
`spikes/genai_external_swap/measure_rss.py`.
"""

from __future__ import annotations

from spikes.genai_external_swap.measure_rss import RssTrace, rss_kb  # noqa: F401

__all__ = ["RssTrace", "rss_kb"]
