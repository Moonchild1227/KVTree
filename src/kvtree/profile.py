"""Replay event logs into reproducible metric and tree materializations."""

from __future__ import annotations

import json
import heapq
import time
from pathlib import Path

from . import collect
from .reprocess import snapshot, to_event
from .storage import atomic_json, event_files


def records(source: Path, stats: dict | None = None):
    def read(path):
        for line in path.open():
            try:
                r = json.loads(line)
            except ValueError:
                if stats is not None: stats["bad_lines"] += 1
                continue
            yield (r.get("recv_ts", 0), r.get("stream", ""), r.get("seq", -1), r)
    streams = [read(p) for p in event_files(source)]
    for _, _, _, row in heapq.merge(*streams):
        yield row


def build(source: Path, out: Path, snapshot_interval: float = 5.0,
          tree_interval: float = 15.0) -> dict:
    if snapshot_interval <= 0 or tree_interval <= 0:
        raise ValueError("profile intervals must be greater than zero")
    out.mkdir(parents=True, exist_ok=True)
    trees = out / "trees"
    trees.mkdir(exist_ok=True)
    kve = collect.load_kv_events_module()
    streams = {}
    next_snap = next_tree = None
    batches = events = 0
    bad = {"bad_lines": 0}
    last_now = time.time()
    with (out / "snapshots.jsonl").open("w") as snap_fp:
        for rec in records(source, bad):
            try:
                now, name = float(rec["recv_ts"]), rec["stream"]
            except (KeyError, TypeError, ValueError):
                bad["bad_lines"] += 1
                continue
            last_now = now
            st = streams.setdefault(name, collect.StreamState(name))
            st.batches += 1
            if rec.get("seq") is not None:
                st.check_seq(int(rec["seq"]))
            for raw_event in rec.get("events") or []:
                st.apply(to_event(raw_event, kve), now, kve)
                events += 1
            batches += 1
            if next_snap is None:
                next_snap = now
                next_tree = now
            while now >= next_snap:
                snap_fp.write(json.dumps(snapshot(streams, next_snap)) + "\n")
                next_snap += snapshot_interval
            while now >= next_tree:
                atomic_json(trees / f"tree_{int(next_tree)}.json", {
                    "ts": next_tree, "streams": {n: s.tree_dump() for n, s in streams.items()}})
                next_tree += tree_interval
        # The final state is useful even when the input is empty.
        final = snapshot(streams, last_now)
        snap_fp.write(json.dumps(final) + "\n")
    payload = {"ts": final["ts"],
               "streams": {n: s.tree_dump() for n, s in streams.items()}}
    atomic_json(out / "live_tree.json", payload)
    result = {"batches": batches, "events": events, "bad_lines": bad["bad_lines"],
              "streams": len(streams), "profile_dir": str(out)}
    atomic_json(out / "profile.json", result)
    return result
