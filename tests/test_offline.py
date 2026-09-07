"""Offline replay: run layout, import, and profile.

The bugs these pin down: `profile` overwriting the run it was reading, snapshots
collapsing onto event timestamps instead of a fixed grid (an idle gap then
disappears from the timeline), a re-import silently doubling every record, and
live capture refusing to resume into its own output directory.
"""

import json
from pathlib import Path

import pytest

from kvtree import cli, reprocess
from kvtree.profile import build, records
from kvtree.storage import RunWriter, import_run, read_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "raw_events.jsonl"
T0 = 1700000000.0


def write_log(path: Path, offsets_s, stream="h:1") -> Path:
    """A one-stream log storing one block per batch at each given offset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        for i, dt in enumerate(offsets_s):
            fp.write(json.dumps({
                "recv_ts": T0 + dt, "stream": stream, "seq": i,
                "batch_ts": T0 + dt, "attn_dp_rank": 0,
                "events": [{"type": "BlockStored", "block_hashes": [1000 + i],
                            "parent_block_hash": 999 + i if i else None,
                            "block_size": 16, "token_ids_len": 16,
                            "medium": "GPU"}]}) + "\n")
    return path


def grid(out: Path) -> list[float]:
    return [round(json.loads(line)["ts"] - T0, 2)
            for line in (out / "snapshots.jsonl").read_text().splitlines()]


def test_snapshots_land_on_a_fixed_grid_across_idle_gaps(tmp_path):
    """Five batches, a 60s tool-wait, five more. The KV stays resident through
    the gap, so the gap has to keep producing snapshots -- emitting one per
    batch instead would erase the plateau the dashboard is meant to show."""
    src = write_log(tmp_path / "run" / "raw_events.jsonl",
                    [0, 1, 2, 3, 4, 64, 65, 66, 67, 68])
    build(src.parent, tmp_path / "pf", snapshot_interval=5.0, tree_interval=15.0)
    assert grid(tmp_path / "pf") == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                     55, 60, 65, 68]
    assert sorted(p.name for p in (tmp_path / "pf" / "trees").iterdir()) == [
        "tree_1700000000.json", "tree_1700000015.json", "tree_1700000030.json",
        "tree_1700000045.json", "tree_1700000060.json"]


def test_profile_and_reprocess_share_one_implementation(tmp_path):
    """Both front ends must produce the same grid and totals; they used to carry
    separate copies of this loop and drifted apart."""
    pf, rp = tmp_path / "pf", tmp_path / "rp"
    result = build(FIXTURE, pf, 0.5, 0.7)
    assert reprocess.main(["--raw", str(FIXTURE), "--out", str(rp),
                           "--snapshot-interval", "0.5",
                           "--tree-dump-interval", "0.7"]) == 0
    assert grid(pf) == grid(rp) == [0.0, 0.5, 1.0, 1.5, 1.7]
    assert [p.name for p in sorted((pf / "trees").iterdir())] == \
           [p.name for p in sorted((rp / "trees").iterdir())]
    assert result["events"] == 36 and result["streams"] == 2
    assert result["totals"] == {"GPU": {"blocks": 24, "tokens": 384},
                                "CPU_PINNED": {"blocks": 2, "tokens": 32}}


def test_profile_refuses_to_overwrite_the_run_it_reads(tmp_path):
    """--out defaulting to --run truncated the recorded snapshots.jsonl of the
    run being profiled, and corrupted it outright if monitor still held it."""
    src = write_log(tmp_path / "run" / "raw_events.jsonl", [0, 1])
    keep = src.parent / "snapshots.jsonl"
    keep.write_text('{"ts": 1, "recorded": true}\n')
    with pytest.raises(ValueError, match="must differ from the input run"):
        build(src.parent, src.parent, 5.0, 15.0)
    assert json.loads(keep.read_text())["recorded"] is True


def test_profile_refuses_an_output_it_already_wrote(tmp_path):
    src = write_log(tmp_path / "run" / "raw_events.jsonl", [0, 1])
    build(src.parent, tmp_path / "pf", 5.0, 15.0)
    with pytest.raises(FileExistsError, match="already exists"):
        build(src.parent, tmp_path / "pf", 5.0, 15.0)
    # reprocess deliberately keeps rewriting the same output dir
    build(src.parent, tmp_path / "pf", 5.0, 15.0, allow_existing=True)


def test_zero_tree_interval_disables_dumps(tmp_path):
    """0 means "off" for `monitor --tree-dump-interval`; profile must not reject
    a flag value carried over from the capture command."""
    src = write_log(tmp_path / "run" / "raw_events.jsonl", [0, 1])
    build(src.parent, tmp_path / "pf", 5.0, 0)
    assert not (tmp_path / "pf" / "trees").exists()
    assert grid(tmp_path / "pf") == [0.0, 1.0]
    with pytest.raises(ValueError, match="snapshot-interval"):
        build(src.parent, tmp_path / "pf2", 0, 15.0)


def test_import_round_trip_matches_profiling_the_raw_log(tmp_path):
    run = tmp_path / "imported"
    manifest = import_run(FIXTURE, run)
    assert manifest["batches"] == 36
    assert sorted(manifest["streams"]) == ["127.0.0.1:37004", "127.0.0.1:37005"]
    assert [p.name for p in sorted(run.rglob("events/**/*.jsonl"))]
    direct = build(FIXTURE, tmp_path / "direct", 0.5, 0.7)
    viaimport = build(run, tmp_path / "viaimport", 0.5, 0.7)
    for key in ("batches", "events", "streams", "totals"):
        assert direct[key] == viaimport[key], key
    assert grid(tmp_path / "direct") == grid(tmp_path / "viaimport")


def test_import_refuses_to_double_an_existing_run(tmp_path):
    run = tmp_path / "imported"
    import_run(FIXTURE, run)
    with pytest.raises(FileExistsError, match="run already exists"):
        import_run(FIXTURE, run)
    lines = sum(len(p.read_text().splitlines())
                for p in run.rglob("events/**/*.jsonl"))
    assert lines == 36, "a second import must not append duplicate records"


def test_import_validates_its_inputs_before_creating_a_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="input does not exist"):
        import_run(tmp_path / "nope.jsonl", tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="turns file does not exist"):
        import_run(FIXTURE, tmp_path / "out2", tmp_path / "nope.jsonl")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out2").exists()


def test_writer_resumes_a_run_but_import_may_not(tmp_path):
    """monitor has always appended to its out-dir, so a restarted capture must
    keep writing into the same run and carry the manifest counters forward.
    An import into an existing run is the case that has to be refused."""
    run = tmp_path / "run"
    rec = {"recv_ts": T0, "stream": "h:1", "seq": 0, "events": []}
    first = RunWriter(run, {"hosts": "a"}, append=True)
    first.append(rec)
    first.close()
    again = RunWriter(run, {"hosts": "a"}, append=True)
    again.append({**rec, "recv_ts": T0 + 1, "seq": 1})
    again.close()
    manifest = read_manifest(run)
    assert manifest["batches"] == 2
    assert manifest["streams"]["h:1"]["batches"] == 2
    assert manifest["started_at"] == first.started, "resume keeps the run's start"
    assert sum(len(p.read_text().splitlines())
               for p in run.rglob("events/**/*.jsonl")) == 2
    with pytest.raises(FileExistsError, match="run already exists"):
        RunWriter(run, {"hosts": "a"})


def test_records_streams_instead_of_materializing(tmp_path):
    """records() reads lazily: a multi-GB run has to replay in memory
    proportional to the shard count, not to the log size."""
    log = write_log(tmp_path / "run" / "raw_events.jsonl", [0, 1])
    with log.open("a") as fp:
        fp.write("{ truncated\n")
    stats = {"bad_lines": 0}
    rows = records(log.parent, stats)
    next(rows)
    assert stats["bad_lines"] == 0, "the trailing bad line was read too early"
    assert len(list(rows)) == 1
    assert stats["bad_lines"] == 1


def test_records_key_survives_null_seq_and_junk(tmp_path):
    """A null seq is valid input everywhere else in the replay path, so it must
    not reach a comparison against an int -- nor may two records ever fall
    through to comparing their dicts."""
    shard = tmp_path / "run" / "events" / "stream=h%3A1"
    shard.mkdir(parents=True)
    rec = {"recv_ts": T0, "stream": "h:1", "seq": None, "events": []}
    (shard / "2023-11-14T22.jsonl").write_text(json.dumps(rec) + "\n")
    (shard / "2023-11-14T23.jsonl").write_text(json.dumps({**rec, "seq": 5}) + "\n")
    assert [r["seq"] for r in records(tmp_path / "run")] == [None, 5]
    stats = {"bad_lines": 0}
    (shard / "2023-11-14T22.jsonl").write_text(
        '[1,2,3]\n{"recv_ts":"x","stream":"h:1","seq":"y","events":[]}\n')
    assert len(list(records(tmp_path / "run", stats))) == 2
    assert stats["bad_lines"] == 1


def test_cli_reports_user_errors_without_a_traceback(tmp_path, capsys):
    run = tmp_path / "run"
    write_log(run / "raw_events.jsonl", [0, 1])
    assert cli.main(["profile", "--run", str(run), "--out", str(run)]) == 2
    err = capsys.readouterr().err
    assert "must differ from the input run" in err
    assert "Traceback" not in err
    assert cli.main(["import", "--raw", str(tmp_path / "nope"),
                     "--out", str(tmp_path / "x")]) == 2
    assert "input does not exist" in capsys.readouterr().err


def test_cli_import_then_profile(tmp_path):
    run, pf = tmp_path / "run", tmp_path / "pf"
    turns = tmp_path / "turns.jsonl"
    turns.write_text('{"session_id": "s1", "turn": 0}\n')
    assert cli.main(["import", "--raw", str(FIXTURE), "--out", str(run),
                     "--turns", str(turns)]) == 0
    # serve finds the copied turns file without being pointed at it
    assert (run / "turns" / "turns.jsonl").read_text() == turns.read_text()
    assert cli.main(["profile", "--run", str(run), "--out", str(pf),
                     "--snapshot-interval", "0.5"]) == 0
    assert json.loads((pf / "profile.json").read_text())["events"] == 36


def test_serve_finds_the_imported_turns_file(tmp_path):
    """`import --turns` used to drop the file somewhere nothing read it."""
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from kvtree.server import make_handler

    run = tmp_path / "run"
    import_run(FIXTURE, run, _turns_file(tmp_path))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(run, None))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        body = urllib.request.urlopen(
            f"http://127.0.0.1:{srv.server_address[1]}/api/turns").read()
    finally:
        srv.shutdown()
    assert [t["session_id"] for t in json.loads(body)] == ["s1"]


def _turns_file(tmp_path: Path) -> Path:
    p = tmp_path / "turns.jsonl"
    p.write_text(json.dumps({"session_id": "s1", "turn": 0, "sent_at_ns": 0,
                             "recv_at_ns": 0, "prompt_tokens": 1,
                             "completion_tokens": 1, "cached_tokens": 0,
                             "client_e2e_ms": 1, "tool_sleep_ms": 0,
                             "http_status": 200, "finish_reason": "stop"}) + "\n")
    return p
