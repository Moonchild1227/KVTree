#!/usr/bin/env python3
"""KV events monitor for sglang (wenxin-sglang) kv-events streams.

Subscribes to one or more kv-events ZMQ PUB streams (one per backend DP rank),
rebuilds the per-stream KV radix tree (block_hash -> parent/medium), and emits:

  - raw_events.jsonl   : every decoded event batch (for offline replay/join)
  - snapshots.jsonl    : per-time-point stats (per stream + global), including
                         L1(GPU)/L2(CPU_PINNED)/L3(EXTERNAL) block & token counts,
                         tree depth / trunk / branching stats, seq-gap counters.
                         L3 is INFERRED: the engine never emits EXTERNAL events,
                         so a block that leaves its last tracked tier (after
                         having been in the host cache, i.e. queued for mooncake
                         backup) is kept as an inferred-EXTERNAL block.
  - live_tree.json     : latest full tree of all streams (refreshed periodically)
  - trees/tree_<ts>.json : per-time-point tree history (for time scrubbing)
  - tree_<stream>_final.json : full tree structure dump at shutdown
  - console live table

Wire protocol mirrors sglang.srt.disaggregation.kv_events:
  multipart frames = [topic, seq(8B big-endian), msgpack(KVEventBatch)]

Usage example:
  python3 kv_events_monitor.py \
      --hosts 127.0.0.1 \
      --base-port 37004 --dp-size 4 --topic kv-events \
      --out-dir ./kvmon_out --snapshot-interval 5 --tree-dump-interval 15

NOTE: if the engine was started without a replay endpoint, this tool must be
running BEFORE the workload starts, otherwise early events are lost and the
reconstructed tree will be incomplete (seq gaps are reported in snapshots).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import msgspec
import zmq
from .storage import RunWriter, atomic_json

# Optional: decode with the engine's own kv_events.py instead of the mirror
# below. Left empty by default -- the mirror is deliberately more permissive.
DEFAULT_KV_EVENTS_PY = ""

MEDIUM_L1 = "GPU"
MEDIUM_L2 = "CPU_PINNED"
MEDIUM_L3 = "EXTERNAL"
MEDIUM_DISK = "DISK"
MEDIUM_UNKNOWN = "UNKNOWN"
MEDIUM_TOMBSTONE = "TOMBSTONE"
# fastest tier first: a block resident in several tiers is reported
# as the fastest one (GPU wins over CPU_PINNED, etc.)
MEDIUM_ORDER = (MEDIUM_L1, MEDIUM_L2, MEDIUM_DISK, MEDIUM_L3)


# ---------------------------------------------------------------------------
# wire protocol: load the real sglang kv_events module (pure-python deps only),
# or use the relaxed local mirror below.
#
# The mirror lives at module scope on purpose: `from __future__ import
# annotations` turns annotations into strings that msgspec resolves against
# module globals, so structs nested in a function cannot be built.
#
# token_ids is an untyped list on purpose. DSv4 emits per-page tokens either
# flat (list[int]) or as (token, next_token) pairs -- see mem_cache/events.py
# lines 66 and 68 -- so the engine's own `list[int]` annotation cannot decode
# its own output ("Expected `int`, got `array`").
# ---------------------------------------------------------------------------
class _KVCacheEvent(msgspec.Struct, array_like=True, gc=False, tag=True):
    pass


class _BlockStored(_KVCacheEvent, tag="BlockStored"):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None


class _BlockRemoved(_KVCacheEvent, tag="BlockRemoved"):
    block_hashes: list[int]
    medium: Optional[str] = None


class _AllBlocksCleared(_KVCacheEvent, tag="AllBlocksCleared"):
    pass


class _KVEventBatch(msgspec.Struct, array_like=True, gc=False):
    ts: float
    events: list[Union[_BlockStored, _BlockRemoved, _AllBlocksCleared]]
    attn_dp_rank: Optional[int] = None


class _MirrorModule:
    BlockStored = _BlockStored
    BlockRemoved = _BlockRemoved
    AllBlocksCleared = _AllBlocksCleared
    KVEventBatch = _KVEventBatch


def load_kv_events_module(path: str | None = None):
    if path is None:
        return _MirrorModule()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"schema file not found: {p}")
    spec = importlib.util.spec_from_file_location("sglang_kv_events", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load schema module: {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# per-stream tree state
# ---------------------------------------------------------------------------
@dataclass
class Block:
    parent: Optional[int]
    media: set          # every tier this block is currently resident in
    tokens: int
    first_seen: float
    children: set = field(default_factory=set)
    # was resident in the host cache at some point, so a mooncake (L3) copy
    # was queued for backup; used to infer L3 residency after the last
    # tree-tracked tier evicts the block
    backed_up: bool = False
    # created to hold a child whose parent hash was never observed (e.g.
    # written before capture started); adopted into the real tree when the
    # parent's BlockStored finally arrives
    placeholder: bool = False
    # structurally present but holds no KV: the engine frees an unbacked
    # device's layers but keeps a node that still has children registered
    # (_delete_unbacked_device_leaf in unified_tree_core.py), so mirror that
    # instead of deleting the node and re-rooting its subtree
    tombstone: bool = False

    @property
    def medium(self) -> str:
        """Highest tier the block lives in.

        With hicache write_through a block is stored to GPU and then to
        CPU_PINNED, so last-write-wins would label everything L2 and hide L1
        entirely. Residency is a set; the tier reported is the fastest one
        present, which is what "which level serves this block" means.
        """
        if self.placeholder:
            return MEDIUM_UNKNOWN
        for m in MEDIUM_ORDER:
            if m in self.media:
                return m
        if self.tombstone:
            return MEDIUM_TOMBSTONE
        return MEDIUM_UNKNOWN


class StreamState:
    """Radix-tree state reconstructed from one kv-events stream."""

    def __init__(self, name: str):
        self.name = name
        self.blocks: dict[int, Block] = {}
        self.last_seq: Optional[int] = None
        self.seq_gaps = 0          # number of missing-batch gaps observed
        self.seq_gap_batches = 0   # total missing batches
        self.batches = 0
        self.events = 0
        self.stored = 0
        self.removed = 0
        self.cleared = 0

    # -- event application --------------------------------------------------
    def check_seq(self, seq: int):
        if self.last_seq is not None and seq > self.last_seq + 1:
            self.seq_gaps += 1
            self.seq_gap_batches += seq - self.last_seq - 1
        self.last_seq = seq

    def apply(self, ev, now: float, kve):
        self.events += 1
        if isinstance(ev, kve.BlockStored):
            self.stored += 1
            parent = ev.parent_block_hash
            n = len(ev.block_hashes)
            per = ev.block_size or (len(ev.token_ids) // n if n else 0)
            for h in ev.block_hashes:
                med = ev.medium or MEDIUM_UNKNOWN
                if parent is not None and parent not in self.blocks:
                    # The parent was never observed (written before capture
                    # started, or its events were lost). Create a structural
                    # placeholder so every child chains onto one shared
                    # unknown trunk instead of dangling as an independent
                    # root; the placeholder is adopted when the parent's
                    # BlockStored finally arrives.
                    self.blocks[parent] = Block(parent=None, media=set(),
                                                tokens=0, first_seen=now,
                                                placeholder=True)
                if h in self.blocks:  # re-store: add residency, keep topology
                    b = self.blocks[h]
                    if b.placeholder:
                        # The real write arrives: adopt the placeholder into
                        # the true tree. Children are already attached to the
                        # node, so the whole subtree reconnects at once.
                        # first_seen stays a lower bound (>= its oldest child).
                        b.placeholder = False
                        b.tombstone = False
                        b.parent = parent
                        b.tokens = per
                        b.media = {med}
                        b.backed_up = med == MEDIUM_L2
                    else:
                        b.media.add(med)
                        if med == MEDIUM_L2:
                            b.backed_up = True
                        if b.tombstone and b.media:
                            b.tombstone = False
                else:
                    self.blocks[h] = Block(
                        parent=parent,
                        media={med},
                        tokens=per,
                        first_seen=now,
                        backed_up=med == MEDIUM_L2,
                    )
                if parent in self.blocks:
                    self.blocks[parent].children.add(h)
                parent = h  # chain: block i+1's parent is block i
        elif isinstance(ev, kve.BlockRemoved):
            self.removed += 1
            med = getattr(ev, "medium", None)
            for h in ev.block_hashes:
                b = self.blocks.get(h)
                if b is None:
                    continue
                if med and med in b.media and len(b.media) > 1:
                    b.media.discard(med)   # still resident in a slower tier
                    continue
                if b.placeholder:
                    # We never saw this block's write, so its residency is
                    # unknown; deleting it would re-break every chain hanging
                    # on it. Keep it grey -- a later re-store adopts it.
                    continue
                # Last tracked residency is gone. A block that passed through
                # the host cache was queued for mooncake backup, so its prefix
                # most likely still lives in L3: keep it as an INFERRED
                # external block instead of dropping it. The engine emits no
                # EXTERNAL events and mooncake-side eviction is invisible, so
                # on long runs this set can only overcount.
                if b.backed_up:
                    b.media = {MEDIUM_L3}
                    b.tombstone = False
                    continue
                if b.children:
                    # The engine frees an unbacked node's device layers but
                    # keeps the node as a structural tombstone while it still
                    # has children (unified_tree_core.py
                    # _delete_unbacked_device_leaf). Mirror that: keep the
                    # structure, drop the KV residency, never re-root.
                    b.media = set()
                    b.tombstone = True
                    continue
                self.blocks.pop(h, None)
                if b.parent in self.blocks:
                    self.blocks[b.parent].children.discard(h)
                # orphaned children are re-rooted (parent pointer kept for dump)
        elif isinstance(ev, kve.AllBlocksCleared):
            self.cleared += 1
            self.blocks.clear()

    # -- stats ---------------------------------------------------------------
    def medium_stats(self) -> dict:
        out = {}
        for b in self.blocks.values():
            s = out.setdefault(b.medium, {"blocks": 0, "tokens": 0})
            s["blocks"] += 1
            s["tokens"] += b.tokens
        return out

    def tree_stats(self) -> dict:
        if not self.blocks:
            return {"roots": 0, "leaves": 0, "max_depth": 0,
                    "mean_leaf_depth": 0.0, "trunk_tokens": 0,
                    "trunk_blocks": 0, "branch_nodes": 0,
                    "placeholder_blocks": 0, "tombstone_blocks": 0}
        roots = [h for h, b in self.blocks.items()
                 if b.parent not in self.blocks]
        leaves = [h for h, b in self.blocks.items() if not b.children]
        branch_nodes = sum(1 for b in self.blocks.values() if len(b.children) > 1)

        # depth of each node via memoized walk to root (iterative)
        depth_cache: dict[int, int] = {}

        def depth(h0: int) -> int:
            h, path = h0, []
            while h is not None and h in self.blocks and h not in depth_cache:
                path.append(h)
                h = self.blocks[h].parent
            base = depth_cache.get(h, 0) if h is not None else 0
            for i, node in enumerate(reversed(path)):
                depth_cache[node] = base + i + 1
            return depth_cache.get(path[0], base) if path else base

        leaf_depths = [depth(h) for h in leaves]
        max_depth = max(leaf_depths) if leaf_depths else 0
        mean_depth = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0.0

        # trunk: longest chain from any root while the node has exactly 1 child
        # (the shared-prefix "main stem"); report the longest one in tokens.
        best_trunk_tokens = 0
        best_trunk_blocks = 0
        for r in roots:
            t, n, h = 0, 0, r
            while h in self.blocks:
                b = self.blocks[h]
                t += b.tokens
                n += 1
                if len(b.children) != 1:
                    break
                h = next(iter(b.children))
            if t > best_trunk_tokens:
                best_trunk_tokens, best_trunk_blocks = t, n

        return {"roots": len(roots), "leaves": len(leaves),
                "max_depth": max_depth,
                "mean_leaf_depth": round(mean_depth, 2),
                "trunk_tokens": best_trunk_tokens,
                "trunk_blocks": best_trunk_blocks,
                "branch_nodes": branch_nodes,
                "placeholder_blocks": sum(1 for b in self.blocks.values()
                                          if b.placeholder),
                "tombstone_blocks": sum(1 for b in self.blocks.values()
                                        if b.tombstone)}

    def tree_dump(self) -> dict:
        """Full nested tree (hashes shortened to 12 hex chars).

        Iterative: chains can be 10k+ blocks deep, recursion would blow up.
        """
        def node(h: int) -> dict:
            root = {"hash": f"{h & 0xFFFFFFFFFFFF:012x}",
                    "medium": self.blocks[h].medium,
                    "tokens": self.blocks[h].tokens, "children": []}
            stack = [(h, root)]
            while stack:
                cur, obj = stack.pop()
                for c in sorted(self.blocks[cur].children):
                    if c not in self.blocks:
                        continue
                    b = self.blocks[c]
                    child = {"hash": f"{c & 0xFFFFFFFFFFFF:012x}",
                             "medium": b.medium, "tokens": b.tokens,
                             "children": []}
                    obj["children"].append(child)
                    stack.append((c, child))
            return root

        roots = sorted(h for h, b in self.blocks.items()
                       if b.parent not in self.blocks)
        return {"stream": self.name, "blocks": len(self.blocks),
                "roots": [node(r) for r in roots]}


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------
class Monitor:
    def __init__(self, args):
        self.args = args
        self.kve = load_kv_events_module(args.schema)
        self.decoder = msgspec.msgpack.Decoder(self.kve.KVEventBatch)
        self.out = Path(args.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trees_dir = self.out / "trees"
        if args.tree_dump_interval > 0:
            self.trees_dir.mkdir(exist_ok=True)
        # Both event sinks are append-mode on purpose: a restarted capture keeps
        # writing into the same run directory.
        #
        # raw_events.jsonl and events/ hold the same lines. The flat file is the
        # documented replay input (`reprocess --raw`) and predates the sharded
        # layout; the shards are what `profile`/`serve` read for a long run.
        # Both are written from one encode -- drop the flat file once nothing
        # points at it any more.
        self.run_writer = RunWriter(self.out, {"hosts": args.hosts,
                                               "base_port": args.base_port,
                                               "dp_size": args.dp_size},
                                    append=True)
        self.raw_fp = open(self.out / "raw_events.jsonl", "a", buffering=1)
        self.snap_fp = open(self.out / "snapshots.jsonl", "a", buffering=1)
        self.streams: dict[str, StreamState] = {}
        self.ctx = zmq.Context.instance()
        self.poller = zmq.Poller()
        self.sockets: dict[zmq.Socket, str] = {}
        self._stop = False
        self._next_tree_dump = time.time() + max(args.tree_dump_interval, 1)

        for ep in self._endpoints():
            s = self.ctx.socket(zmq.SUB)
            s.set_hwm(args.sub_hwm)
            s.connect(ep["addr"])
            s.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
            name = ep["name"]
            self.sockets[s] = name
            self.streams[name] = StreamState(name)
            self.poller.register(s, zmq.POLLIN)
            print(f"[monitor] subscribed {name} -> {ep['addr']}", flush=True)

    def _endpoints(self) -> list[dict]:
        eps = []
        for host in self.args.hosts.split(","):
            host = host.strip()
            for dp in range(self.args.dp_size):
                port = self.args.base_port + dp
                eps.append({"name": f"{host}:{port}",
                            "addr": f"tcp://{host}:{port}"})
        return eps

    def stop(self, *_):
        self._stop = True

    def run(self):
        next_snap = time.time() + self.args.snapshot_interval
        deadline = (time.time() + self.args.duration) if self.args.duration else None
        while not self._stop:
            timeout = max(0.0, min(1.0, next_snap - time.time()))
            events = dict(self.poller.poll(timeout * 1000))
            now = time.time()
            for sock in events:
                self._drain(sock, now)
            if now >= next_snap:
                self.snapshot(now)
                next_snap = now + self.args.snapshot_interval
            if self.args.tree_dump_interval > 0 and now >= self._next_tree_dump:
                self.dump_trees(now)
                self._next_tree_dump = now + self.args.tree_dump_interval
            if deadline and now >= deadline:
                break
        self.finish()

    def _drain(self, sock, now):
        name = self.sockets[sock]
        st = self.streams[name]
        while True:
            try:
                frames = sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            if len(frames) != 3:
                continue
            _, seq_b, payload = frames
            seq = int.from_bytes(seq_b, "big")
            st.check_seq(seq)
            try:
                batch = self.decoder.decode(payload)
            except Exception as e:  # keep going on bad payload
                print(f"[monitor] decode error on {name}: {e}", flush=True)
                continue
            st.batches += 1
            rec = {"recv_ts": now, "stream": name, "seq": seq,
                   "batch_ts": batch.ts,
                   "attn_dp_rank": batch.attn_dp_rank,
                   "events": [self._event_to_json(e) for e in batch.events]}
            encoded = json.dumps(rec) + "\n"
            self.raw_fp.write(encoded)
            self.run_writer.append(rec, encoded)
            for e in batch.events:
                st.apply(e, now, self.kve)

    @staticmethod
    def _event_to_json(e) -> dict:
        d = {"type": type(e).__name__}
        for f in e.__struct_fields__:
            v = getattr(e, f)
            if f == "token_ids":  # keep raw file small; hashes are the key
                d["token_ids_len"] = len(v) if v is not None else 0
            else:
                d[f] = v
        return d

    # -- snapshots ------------------------------------------------------------
    def snapshot(self, now: float):
        per_stream = {}
        totals: dict[str, dict] = {}
        for name, st in self.streams.items():
            ms = st.medium_stats()
            ts = st.tree_stats()
            per_stream[name] = {
                "mediums": ms, "tree": ts,
                "batches": st.batches, "events": st.events,
                "stored": st.stored, "removed": st.removed,
                "cleared": st.cleared,
                "seq_gaps": st.seq_gaps,
                "seq_gap_batches": st.seq_gap_batches,
                "last_seq": st.last_seq,
            }
            for med, v in ms.items():
                t = totals.setdefault(med, {"blocks": 0, "tokens": 0})
                t["blocks"] += v["blocks"]
                t["tokens"] += v["tokens"]
        rec = {"ts": now, "time": time.strftime("%H:%M:%S", time.localtime(now)),
               "totals": totals, "streams": per_stream}
        self.snap_fp.write(json.dumps(rec) + "\n")
        self._print(rec)

    def _print(self, rec):
        t = rec["totals"]

        def tok(m):
            v = t.get(m, {}).get("tokens", 0)
            return f"{v/1024:8.1f}k" if v < 10**7 else f"{v/1048576:8.2f}M"

        nblocks = sum(v["blocks"] for v in t.values())
        print(f"[{rec['time']}] blocks={nblocks:6d} | "
              f"L1(GPU)={tok(MEDIUM_L1)} tok  "
              f"L2(CPU)={tok(MEDIUM_L2)} tok  "
              f"L3(EXT)={tok(MEDIUM_L3)} tok", flush=True)

    # -- tree dumps -------------------------------------------------------------
    def dump_trees(self, now: float):
        payload = {"ts": now,
                   "time": time.strftime("%H:%M:%S", time.localtime(now)),
                   "streams": {n: st.tree_dump()
                               for n, st in self.streams.items()
                               if st.blocks}}
        atomic_json(self.out / "live_tree.json", payload)
        if self.args.tree_dump_interval > 0:
            atomic_json(self.trees_dir / f"tree_{int(now)}.json", payload)

    # -- shutdown ---------------------------------------------------------------
    def finish(self):
        self.snapshot(time.time())
        self.dump_trees(time.time())
        for name, st in self.streams.items():
            if st.blocks:
                p = self.out / f"tree_{name.replace(':', '_')}_final.json"
                atomic_json(p, st.tree_dump())
        self.raw_fp.close()
        self.snap_fp.close()
        self.run_writer.close()
        print(f"[monitor] done. outputs in {self.out}/", flush=True)
