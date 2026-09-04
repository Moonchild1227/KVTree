"""Regression tests for the three bugs that cost the most time to find."""

import json
from pathlib import Path

import msgspec
import pytest

from kvtree import collect
from kvtree.reprocess import to_event

FIXTURE = Path(__file__).parent / "fixtures" / "raw_events.jsonl"


def replay(path=FIXTURE):
    kve = collect.load_kv_events_module("", relaxed=True)
    streams = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        st = streams.setdefault(r["stream"], collect.StreamState(r["stream"]))
        st.batches += 1
        if r.get("seq") is not None:
            st.check_seq(int(r["seq"]))
        for e in r["events"]:
            st.apply(to_event(e, kve), r["recv_ts"], kve)
    return streams, kve


def test_paired_token_ids_decode():
    """DSv4 emits per-page tokens as (tok, next) pairs; the engine's own
    `token_ids: list[int]` annotation cannot decode that. The mirror must."""
    kve = collect.load_kv_events_module("", relaxed=True)
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(kve.KVEventBatch)
    for tokens in ([(1, 2), (3, 4)], [1, 2, 3, 4]):
        batch = kve.KVEventBatch(ts=1.0, attn_dp_rank=0, events=[
            kve.BlockStored(block_hashes=[7], parent_block_hash=None,
                            token_ids=tokens, block_size=2, lora_id=None,
                            medium="GPU")])
        got = dec.decode(enc.encode(batch)).events[0]
        assert got.block_hashes == [7]
        assert len(got.token_ids) == len(tokens)


def test_write_through_counts_as_gpu():
    """A block stored to GPU and then mirrored to CPU_PINNED is served from
    GPU. Last-write-wins would report the whole tree as L2 and hide L1."""
    streams, _ = replay()
    st = next(iter(streams.values()))
    tiers = st.medium_stats()
    assert "GPU" in tiers, tiers
    multi = [b for b in st.blocks.values() if len(b.media) > 1]
    assert multi, "fixture should contain dual-resident blocks"
    assert all(b.medium == "GPU" for b in multi)


def test_per_medium_removal_keeps_slower_tier():
    """BlockRemoved carries a medium: evicting from GPU must leave the block
    resident in CPU_PINNED rather than deleting the node."""
    streams, _ = replay()
    st = next(iter(streams.values()))
    survivors = [b for b in st.blocks.values() if b.media == {"CPU_PINNED"}]
    assert survivors, "GPU-only eviction should leave a CPU_PINNED block"
    assert survivors[0].medium == "CPU_PINNED"


def test_tree_topology_and_stats():
    streams, _ = replay()
    st = next(iter(streams.values()))
    stats = st.tree_stats()
    assert stats["roots"] >= 1
    assert stats["leaves"] == 3, stats          # three session branches
    assert stats["max_depth"] >= 6
    dump = st.tree_dump()
    assert dump["blocks"] == len(st.blocks)
    assert st.seq_gaps == 0


def test_two_streams_are_independent():
    streams, _ = replay()
    assert len(streams) == 2
    a, b = (set(s.blocks) for s in streams.values())
    assert a and b and a != b
