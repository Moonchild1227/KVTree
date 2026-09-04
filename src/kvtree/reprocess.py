"""Rebuild snapshots/tree dumps from a captured raw_events.jsonl.

raw_events.jsonl keeps every decoded event, so a change in how the tree state
is derived does not require re-running the engine -- the run is replayable.
Used when the tier-residency rule changed (a block written through GPU->CPU
must count as GPU, not CPU).

  ./reprocess.py --raw <run>/kvmon/raw_events.jsonl --out <run>/kvmon_v2 \
      [--snapshot-interval 5] [--tree-dump-interval 15]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import collect as m


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

    kve = m.load_kv_events_module("/nonexistent", relaxed=True)
    out = Path(a.out)
    (out / "trees").mkdir(parents=True, exist_ok=True)
    snap_fp = (out / "snapshots.jsonl").open("w")

    streams: dict[str, object] = {}
    next_snap = next_tree = None
    n_ev = 0
    for line in Path(a.raw).open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        now = r["recv_ts"]
        st = streams.get(r["stream"])
        if st is None:
            st = streams[r["stream"]] = m.StreamState(r["stream"])
        st.batches += 1
        if r.get("seq") is not None:
            st.check_seq(int(r["seq"]))
        for e in r.get("events") or []:
            st.apply(to_event(e, kve), now, kve)
            n_ev += 1
        if next_snap is None:
            next_snap, next_tree = now, now
        while now >= next_snap:
            snap_fp.write(json.dumps(snapshot(streams, next_snap)) + "\n")
            next_snap += a.snap_int
        while now >= next_tree:
            payload = {"ts": next_tree,
                       "streams": {n: s.tree_dump() for n, s in streams.items()}}
            (out / "trees" / f"tree_{int(next_tree)}.json").write_text(
                json.dumps(payload))
            next_tree += a.tree_int

    last = snapshot(streams, next_snap or time.time())
    snap_fp.write(json.dumps(last) + "\n")
    snap_fp.close()
    payload = {"ts": last["ts"],
               "streams": {n: s.tree_dump() for n, s in streams.items()}}
    (out / "live_tree.json").write_text(json.dumps(payload))
    for n, s in streams.items():
        (out / f"tree_{n.replace(':', '_')}_final.json").write_text(
            json.dumps(s.tree_dump()))
    tot = last["totals"]
    print(f"[reprocess] {n_ev} events -> {out}")
    for med in sorted(tot):
        print(f"  {med:12s} blocks={tot[med]['blocks']:6d} "
              f"tokens={tot[med]['tokens']/1000:.0f}k")
    return 0
