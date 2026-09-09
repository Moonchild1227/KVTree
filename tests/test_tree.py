"""Regression tests for the three bugs that cost the most time to find."""

import json
from pathlib import Path

import msgspec
import pytest

from kvtree import cli, collect
from kvtree.reprocess import to_event

FIXTURE = Path(__file__).parent / "fixtures" / "raw_events.jsonl"


def replay(path=FIXTURE):
    kve = collect.load_kv_events_module()
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
    kve = collect.load_kv_events_module()
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


def test_monitor_cli_finishes_once(monkeypatch, tmp_path):
    calls = {"run": 0, "finish": 0}

    class FakeMonitor:
        def __init__(self, args):
            assert args.schema is None

        def stop(self, *_):
            pass

        def run(self):
            calls["run"] += 1
            self.finish()

        def finish(self):
            calls["finish"] += 1

    monkeypatch.setattr(collect, "Monitor", FakeMonitor)
    args = cli.build_parser().parse_args([
        "monitor", "--out-dir", str(tmp_path), "--duration", "0.01",
    ])
    assert args.fn(args) == 0
    assert calls == {"run": 1, "finish": 1}


def test_explicit_schema_path(tmp_path):
    schema = tmp_path / "kv_events.py"
    schema.write_text("SENTINEL = 42\n")
    module = collect.load_kv_events_module(str(schema))
    assert module.SENTINEL == 42


def test_missing_schema_path_fails_clearly(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError, match="schema file not found"):
        collect.load_kv_events_module(str(missing))


def _store(kve, h, parent, medium="GPU", size=2):
    return kve.BlockStored(block_hashes=[h], parent_block_hash=parent,
                           token_ids=[], block_size=size, lora_id=None,
                           medium=medium)


def _remove(kve, h, medium="GPU"):
    return kve.BlockRemoved(block_hashes=[h], medium=medium)


def test_unknown_parent_gets_placeholder():
    """A store whose parent was never observed must chain onto a shared
    placeholder trunk instead of dangling as an independent root."""
    kve = collect.load_kv_events_module()
    st = collect.StreamState("s")
    st.apply(_store(kve, 11, 999), 1.0, kve)
    assert st.blocks[999].placeholder
    assert st.blocks[11].parent == 999
    stats = st.tree_stats()
    assert stats["roots"] == 1          # the placeholder, not the child
    assert stats["placeholder_blocks"] == 1


def test_placeholder_adopted_when_parent_store_arrives():
    """When the missing parent's real store arrives later, the placeholder
    becomes a real node and the whole subtree reconnects to the true tree."""
    kve = collect.load_kv_events_module()
    st = collect.StreamState("s")
    st.apply(_store(kve, 11, 999), 1.0, kve)      # suffix on unknown parent
    st.apply(_store(kve, 999, 5), 2.0, kve)       # the parent's real write
    st.apply(_store(kve, 5, None), 3.0, kve)      # grandparent at tree root
    assert not st.blocks[999].placeholder
    assert st.blocks[999].parent == 5
    roots = [h for h, b in st.blocks.items() if b.parent not in st.blocks]
    assert roots == [5]
    dump = st.tree_dump()
    top = dump["roots"][0]
    assert top["hash"] == f"{5 & 0xFFFFFFFFFFFF:012x}"
    assert top["children"][0]["children"][0]["hash"] == \
        f"{11 & 0xFFFFFFFFFFFF:012x}"


def test_unbacked_internal_node_becomes_tombstone():
    """Engine semantics (_delete_unbacked_device_leaf): an unbacked node with
    children is NOT deleted -- it stays as a structural tombstone and the
    subtree stays attached."""
    kve = collect.load_kv_events_module()
    st = collect.StreamState("s")
    st.apply(_store(kve, 1, None), 1.0, kve)
    st.apply(_store(kve, 2, 1), 1.1, kve)
    st.apply(_store(kve, 3, 2), 1.2, kve)
    st.apply(_remove(kve, 2), 2.0, kve)           # 2 is unbacked, has child 3
    assert 2 in st.blocks
    assert st.blocks[2].tombstone
    assert st.blocks[2].medium == "TOMBSTONE"
    assert 3 in st.blocks[2].children             # subtree NOT re-rooted
    stats = st.tree_stats()
    assert stats["roots"] == 1 and stats["tombstone_blocks"] == 1


def test_tombstone_refilled_by_restore():
    kve = collect.load_kv_events_module()
    st = collect.StreamState("s")
    st.apply(_store(kve, 1, None), 1.0, kve)
    st.apply(_store(kve, 2, 1), 1.1, kve)
    st.apply(_remove(kve, 1), 2.0, kve)           # 1 becomes tombstone
    st.apply(_store(kve, 1, None), 3.0, kve)      # recomputed -> KV back
    assert not st.blocks[1].tombstone
    assert st.blocks[1].medium == "GPU"


def test_childless_unbacked_block_is_deleted():
    kve = collect.load_kv_events_module()
    st = collect.StreamState("s")
    st.apply(_store(kve, 1, None), 1.0, kve)
    st.apply(_store(kve, 2, 1), 1.1, kve)
    st.apply(_remove(kve, 2), 2.0, kve)           # childless leaf -> gone
    assert 2 not in st.blocks
    assert 1 in st.blocks and not st.blocks[1].tombstone


def test_backed_block_survives_as_inferred_l3():
    """Regression: a block that passed through host cache is kept as inferred
    EXTERNAL when its last tracked tier evicts it."""
    kve = collect.load_kv_events_module()
    st = collect.StreamState("s")
    st.apply(_store(kve, 1, None), 1.0, kve)
    st.apply(_store(kve, 1, None, medium="CPU_PINNED"), 1.5, kve)
    st.apply(_remove(kve, 1, medium="GPU"), 2.0, kve)    # demote GPU copy
    assert st.blocks[1].medium == "CPU_PINNED"
    st.apply(_remove(kve, 1, medium="CPU_PINNED"), 3.0, kve)  # last tier out
    assert st.blocks[1].medium == "EXTERNAL"


def test_make_server_ephemeral_port(tmp_path):
    """Port 0 must bind an ephemeral port and still serve (dashboard fallback
    path when the requested port is busy)."""
    from kvtree.server import make_server
    srv = make_server(tmp_path, None, 0, "127.0.0.1")
    try:
        assert srv.server_address[1] > 0
    finally:
        srv.server_close()
