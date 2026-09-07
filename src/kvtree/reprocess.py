"""Event -> tree-state derivation shared by live collection and offline replay.

`to_event` rebuilds a kv-events struct from a recorded JSON line and `snapshot`
turns the per-stream states into one time point; `profile.build` drives both.
The `reprocess` CLI is the thin legacy front end: a recorded run keeps every
decoded event, so changing how tree state is derived does not require re-running
the engine. Used when the tier-residency rule changed (a block written through
GPU->CPU must count as GPU, not CPU).

  kvtree reprocess --raw <run>/kvmon/raw_events.jsonl --out <run>/kvmon_v2 \
      [--snapshot-interval 5] [--tree-dump-interval 15]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def to_event(e: dict, kve):
    t = e.get("type", "")
    if "BlockStored" in t:
        n = max(1, len(e.get("block_hashes") or [1]))
        return kve.BlockStored(
            block_hashes=e.get("block_hashes") or [],
            parent_block_hash=e.get("parent_block_hash"),
            token_ids=[],
            block_size=e.get("block_size") or (e.get("token_ids_len", 0) // n),
            lora_id=None,
            medium=e.get("medium"),
        )
    if "BlockRemoved" in t:
        return kve.BlockRemoved(block_hashes=e.get("block_hashes") or [],
                                medium=e.get("medium"))
    return kve.AllBlocksCleared()


def snapshot(streams, now: float) -> dict:
    per, totals = {}, {}
    for name, st in streams.items():
        ms, ts = st.medium_stats(), st.tree_stats()
        per[name] = {
            "mediums": ms, "tree": ts,
            "batches": st.batches, "events": st.events,
            "stored": st.stored, "removed": st.removed, "cleared": st.cleared,
            "seq_gaps": st.seq_gaps, "seq_gap_batches": st.seq_gap_batches,
            "last_seq": st.last_seq,
        }
        for med, v in ms.items():
            t = totals.setdefault(med, {"blocks": 0, "tokens": 0})
            t["blocks"] += v["blocks"]
            t["tokens"] += v["tokens"]
    return {"ts": now,
            "time": time.strftime("%H:%M:%S", time.localtime(now)),
            "totals": totals, "streams": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-interval", type=float, default=5.0,
                    dest="snap_int")
    ap.add_argument("--tree-dump-interval", type=float, default=15.0,
                    dest="tree_int")
    a = ap.parse_args(argv)

    # One replay implementation, shared with `kvtree profile` -- the two used to
    # carry separate copies of the snapshot grid and drifted apart.
    from .profile import build
    result = build(Path(a.raw), Path(a.out), a.snap_int, a.tree_int,
                   allow_existing=True)
    print(f"[reprocess] {result['events']} events -> {a.out}")
    tot = result["totals"]
    for med in sorted(tot):
        print(f"  {med:12s} blocks={tot[med]['blocks']:6d} "
              f"tokens={tot[med]['tokens']/1000:.0f}k")
    return 0
