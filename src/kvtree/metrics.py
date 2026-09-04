"""Sample SGLang's /metrics pool gauges next to the kv-events tree.

The radix tree tells you which blocks are *resident*; it cannot tell you which
of them the current batch actually references. The engine's own gauges close
that gap, per attn-dp rank:

  sglang:num_used_tokens      tokens referenced by running requests
  sglang:kv_evictable_tokens  resident but unreferenced == idle L1 cache
  sglang:kv_available_tokens  free pool space
  sglang:token_usage          used fraction of the pool
  sglang:cache_hit_rate       prefix reuse as the engine sees it

Writes one JSON object per poll to metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import time
import urllib.request
from pathlib import Path

WANTED = (
    "num_used_tokens", "kv_evictable_tokens", "kv_available_tokens",
    "token_usage", "cache_hit_rate", "num_running_reqs", "num_queue_reqs",
    "hicache_backup_tokens_total", "hicache_dropped_tokens_total",
)
LINE = re.compile(r"^sglang:(\w+)\{([^}]*)\}\s+([0-9.eE+-]+)$")

_stop = False


def _sig(*_):
    global _stop
    _stop = True


def labels(raw: str) -> dict:
    out = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def scrape(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        body = r.read().decode("utf-8", "replace")
    per_rank: dict[str, dict] = {}
    for line in body.splitlines():
        if not line.startswith("sglang:"):
            continue
        m = LINE.match(line)
        if not m:
            continue
        name, lab, val = m.group(1), labels(m.group(2)), float(m.group(3))
        if name not in WANTED:
            continue
        rank = lab.get("dp_rank") or lab.get("attn_dp_rank") or "0"
        per_rank.setdefault(rank, {})[name] = val
    return per_rank


def run(url: str, out: Path, interval: float = 5.0,
        duration: float = 0.0) -> int:
    """Poll the engine's /metrics and append one JSON object per sample."""
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    out.mkdir(parents=True, exist_ok=True)
    fp = (out / "metrics.jsonl").open("a", buffering=1)
    t0, n = time.time(), 0
    print(f"[metrics] {url} every {interval}s -> {out/'metrics.jsonl'}",
          flush=True)
    while not _stop:
        now = time.time()
        try:
            per_rank = scrape(url)
        except Exception as exc:
            per_rank = {}
            print(f"[metrics] scrape failed: {type(exc).__name__}: {exc}",
                  flush=True)
        agg: dict[str, float] = {}
        for r in per_rank.values():
            for k, v in r.items():
                if k in ("token_usage", "cache_hit_rate"):
                    agg[k] = agg.get(k, 0.0) + v / max(1, len(per_rank))
                else:
                    agg[k] = agg.get(k, 0.0) + v
        fp.write(json.dumps({
            "ts": now,
            "time": time.strftime("%H:%M:%S", time.localtime(now)),
            "ranks": per_rank, "total": agg,
        }) + "\n")
        n += 1
        if n % 12 == 1:
            print(f"[metrics] used={agg.get('num_used_tokens', 0)/1000:.0f}k "
                  f"idle={agg.get('kv_evictable_tokens', 0)/1000:.0f}k "
                  f"free={agg.get('kv_available_tokens', 0)/1000:.0f}k "
                  f"usage={agg.get('token_usage', 0):.3f} "
                  f"running={agg.get('num_running_reqs', 0):.0f}", flush=True)
        if duration and now - t0 >= duration:
            break
        time.sleep(max(0.0, interval - (time.time() - now)))
    fp.close()
    print(f"[metrics] done, {n} samples", flush=True)
    return 0
