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
    """Yield every recorded batch in ``(recv_ts, stream, seq)`` order.

    Streaming, not sorted: shards are merged lazily so a multi-GB run replays in
    memory proportional to the shard count. That means each shard must already
    be ordered -- true for anything monitor wrote (see `storage.event_files`).
    A hand-assembled log that is out of order is replayed as it lies.
    """
    def read(path, shard: int):
        with path.open() as fp:
            for n, line in enumerate(fp):
                try:
                    row = json.loads(line)
                except ValueError:
                    if stats is not None:
                        stats["bad_lines"] += 1
                    continue
                if not isinstance(row, dict):
                    if stats is not None:
                        stats["bad_lines"] += 1
                    continue
                # (shard, n) keeps the key totally ordered on well-formed input
                # too: seq is allowed to be null and two dicts do not compare,
                # so neither may ever reach the comparison.
                yield (_num(row.get("recv_ts")), str(row.get("stream", "")),
                       _num(row.get("seq"), -1), shard, n, row)

    streams = [read(p, i) for i, p in enumerate(event_files(source))]
    for item in heapq.merge(*streams):
        yield item[-1]


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build(source: Path, out: Path, snapshot_interval: float = 5.0,
          tree_interval: float = 15.0, allow_existing: bool = False) -> dict:
    """Replay ``source`` into snapshots and tree dumps under ``out``.

    ``tree_interval <= 0`` disables the periodic tree dumps, matching what the
    flag means for `kvtree monitor`.
    """
    if snapshot_interval <= 0:
        raise ValueError("--snapshot-interval must be greater than zero")
    if not source.exists():
        raise FileNotFoundError(f"input does not exist: {source}")
    if out.resolve() == source.resolve():
        raise ValueError(f"output must differ from the input run: {source}")
    if not allow_existing and (out / "profile.json").exists():
        raise FileExistsError(f"profile output already exists: {out}")
    dump_trees = tree_interval > 0
    out.mkdir(parents=True, exist_ok=True)
    trees = out / "trees"
    if dump_trees:
        trees.mkdir(exist_ok=True)
    kve = collect.load_kv_events_module()
    streams: dict[str, collect.StreamState] = {}
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
            if next_snap is not None:
                # Grid points strictly before this batch describe the state
                # before its events. This preserves resident plateaus during
                # long idle gaps and avoids applying removals retroactively.
                while next_snap < now:
                    snap_fp.write(json.dumps(snapshot(streams, next_snap)) + "\n")
                    next_snap += snapshot_interval
                while dump_trees and next_tree < now:
                    _dump(trees / f"tree_{tree_key(next_tree, tree_interval)}.json",
                          streams, next_tree)
                    next_tree += tree_interval
            st = streams.get(name)
            if st is None:
                st = streams[name] = collect.StreamState(name)
            st.batches += 1
            if rec.get("seq") is not None:
                st.check_seq(int(rec["seq"]))
            for raw_event in rec.get("events") or []:
                st.apply(to_event(raw_event, kve), now, kve)
                events += 1
            batches += 1
            if next_snap is None:
                next_snap = next_tree = now
            # Include the current batch in points exactly at its timestamp.
            while next_snap is not None and now >= next_snap:
                snap_fp.write(json.dumps(snapshot(streams, next_snap)) + "\n")
                next_snap += snapshot_interval
            while dump_trees and next_tree is not None and now >= next_tree:
                _dump(trees / f"tree_{tree_key(next_tree, tree_interval)}.json",
                      streams, next_tree)
                next_tree += tree_interval
        # The final state is useful even when the input is empty.
        final = snapshot(streams, last_now)
        snap_fp.write(json.dumps(final) + "\n")
    _dump(out / "live_tree.json", streams, final["ts"])
    for name, st in streams.items():
        atomic_json(out / f"tree_{name.replace(':', '_')}_final.json",
                    st.tree_dump())
    result = {"batches": batches, "events": events, "bad_lines": bad["bad_lines"],
              "streams": len(streams), "totals": final["totals"],
              "profile_dir": str(out)}
    atomic_json(out / "profile.json", result)
    return result


def _dump(path: Path, streams: dict, ts: float) -> None:
    atomic_json(path, {"ts": ts, "time": time.strftime("%H:%M:%S",
                                                       time.localtime(ts)),
                       "streams": {n: s.tree_dump() for n, s in streams.items()}})


def tree_key(ts: float, interval: float) -> int:
    """Use legacy second keys for normal intervals, milliseconds below one sec."""
    return int(ts if interval >= 1 else ts * 1000)
