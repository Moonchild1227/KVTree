"""kvtree command line entry point."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from . import __version__

# Session files are not auto-discovered on purpose: the workload client owns
# them and may be anywhere. Explicit --turns wins, then this shared variable.
TURNS_ENV = "KVTREE_TURNS"


def resolve_turns(explicit: str | None) -> str | None:
    return explicit or os.environ.get(TURNS_ENV)


def _monitor(a) -> int:
    from . import collect
    args = argparse.Namespace(
        hosts=a.hosts, base_port=a.base_port, dp_size=a.dp_size,
        topic=a.topic, out_dir=a.out_dir, snapshot_interval=a.snapshot_interval,
        tree_dump_interval=a.tree_dump_interval, duration=a.duration,
        schema=a.schema, sub_hwm=a.sub_hwm,
    )
    mon = collect.Monitor(args)
    signal.signal(signal.SIGINT, mon.stop)
    signal.signal(signal.SIGTERM, mon.stop)
    mon.run()
    return 0


def _serve(a) -> int:
    from .server import serve
    serve(Path(a.dir), resolve_turns(a.turns), a.port, a.bind)
    return 0


def _observe(a) -> int:
    """Run monitor + metrics + dashboard in one process.

    kvtree observes, it does not launch SGLang: start this (or monitor)
    before the engine/workload so no kv-events are missed. One Ctrl-C stops
    all three.
    """
    import threading

    from . import collect, metrics
    from .server import make_server

    mon_args = argparse.Namespace(
        hosts=a.hosts, base_port=a.base_port, dp_size=a.dp_size,
        topic=a.topic, out_dir=a.out_dir,
        snapshot_interval=a.snapshot_interval,
        tree_dump_interval=a.tree_dump_interval, duration=0.0,
        schema=a.schema, sub_hwm=a.sub_hwm,
    )
    mon = collect.Monitor(mon_args)

    metrics_stop = threading.Event()
    threads = [
        threading.Thread(target=mon.run, name="kvtree-monitor", daemon=True),
        threading.Thread(
            target=metrics.run, name="kvtree-metrics", daemon=True,
            kwargs=dict(url=a.metrics_url, out=Path(a.out_dir),
                        interval=a.metrics_interval,
                        stop=metrics_stop.is_set)),
    ]

    srv = make_server(Path(a.out_dir), resolve_turns(a.turns), a.port, a.bind)

    stopping = threading.Event()

    def shutdown(*_):
        if not stopping.is_set():
            stopping.set()
            print("[kvtree] shutting down...", flush=True)
        mon.stop()
        metrics_stop.set()
        # serve_forever and shutdown must not run on the same thread
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for t in threads:
        t.start()
    try:
        srv.serve_forever()
    finally:
        shutdown()
        for t in threads:
            t.join(timeout=5)
        srv.server_close()
    return 0


def _metrics(a) -> int:
    from . import metrics
    return metrics.run(url=a.url, out=Path(a.out), interval=a.interval,
                       duration=a.duration)


def _reprocess(a) -> int:
    from . import reprocess
    return reprocess.main([
        "--raw", a.raw, "--out", a.out,
        "--snapshot-interval", str(a.snapshot_interval),
        "--tree-dump-interval", str(a.tree_dump_interval),
    ])


def _import(a) -> int:
    from .storage import import_run
    result = import_run(Path(a.raw), Path(a.out), Path(a.turns) if a.turns else None)
    bad = result.get("bad_lines") or 0
    print(f"[import] {result['batches']} batches -> {a.out}"
          + (f" ({bad} unusable lines skipped)" if bad else ""))
    return 0


def _profile(a) -> int:
    from .profile import build
    result = build(Path(a.run), Path(a.out), a.snapshot_interval,
                   a.tree_dump_interval)
    bad = result.get("bad_lines") or 0
    print(f"[profile] {result['events']} events -> {result['profile_dir']}"
          + (f" ({bad} unusable lines skipped)" if bad else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="kvtree",
        description="Live KV radix tree observability for SGLang.")
    ap.add_argument("--version", action="version", version=f"kvtree {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("monitor", help="subscribe to kv-events and record")
    m.add_argument("--hosts", default="127.0.0.1",
                   help="comma separated engine hosts (default: 127.0.0.1)")
    m.add_argument("--base-port", type=int, default=37004,
                   help="kv-events base port; rank N publishes on base+N")
    m.add_argument("--dp-size", type=int, default=1,
                   help="attn-dp ranks per host")
    m.add_argument("--topic", default="kv-events")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--snapshot-interval", type=float, default=5.0)
    m.add_argument("--tree-dump-interval", type=float, default=15.0)
    m.add_argument("--duration", type=float, default=0.0)
    m.add_argument("--schema", metavar="PATH",
                   help="strictly decode with the specified kv_events.py; "
                        "default: use the permissive built-in schema")
    m.add_argument("--sub-hwm", type=int, default=0,
                   help="ZMQ SUB high-water mark; 0 = unlimited (never drop "
                        "events, at the cost of memory under a burst)")
    m.set_defaults(fn=_monitor)

    s = sub.add_parser("serve", help="serve the dashboard over a data dir")
    s.add_argument("--dir", required=True, help="monitor --out-dir")
    s.add_argument("--turns", default=None,
                   help=f"turns.jsonl for the session panels; also picked up "
                        f"from ${TURNS_ENV}")
    s.add_argument("--port", type=int, default=8899)
    s.add_argument("--bind", default="0.0.0.0")
    s.set_defaults(fn=_serve)

    g = sub.add_parser("metrics", help="sample the engine's /metrics pool gauges")
    g.add_argument("--url", default="http://127.0.0.1:37000/metrics")
    g.add_argument("--out", required=True)
    g.add_argument("--interval", type=float, default=5.0)
    g.add_argument("--duration", type=float, default=0.0)
    g.set_defaults(fn=_metrics)

    o = sub.add_parser("observe",
                       help="run monitor + metrics + dashboard in one process")
    o.add_argument("--hosts", default="127.0.0.1",
                   help="comma separated engine hosts (default: 127.0.0.1)")
    o.add_argument("--base-port", type=int, default=37004,
                   help="kv-events base port; rank N publishes on base+N")
    o.add_argument("--dp-size", type=int, default=1,
                   help="attn-dp ranks per host")
    o.add_argument("--topic", default="kv-events")
    o.add_argument("--out-dir", required=True)
    o.add_argument("--snapshot-interval", type=float, default=5.0)
    o.add_argument("--tree-dump-interval", type=float, default=15.0,
                   help="0 disables the periodic tree dumps")
    o.add_argument("--schema", metavar="PATH",
                   help="strictly decode with the specified kv_events.py; "
                        "default: use the permissive built-in schema")
    o.add_argument("--sub-hwm", type=int, default=0,
                   help="ZMQ SUB high-water mark; 0 = unlimited")
    o.add_argument("--metrics-url",
                   default="http://127.0.0.1:37000/metrics",
                   help="engine /metrics endpoint for the L1 Pool panel")
    o.add_argument("--metrics-interval", type=float, default=5.0)
    o.add_argument("--turns", default=None,
                   help=f"turns.jsonl for the session panels; also picked up "
                        f"from ${TURNS_ENV}")
    o.add_argument("--port", type=int, default=8899)
    o.add_argument("--bind", default="0.0.0.0")
    o.set_defaults(fn=_observe)

    r = sub.add_parser("reprocess",
                       help="rebuild snapshots/trees from recorded raw events")
    r.add_argument("--raw", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--snapshot-interval", type=float, default=5.0)
    r.add_argument("--tree-dump-interval", type=float, default=15.0)
    r.set_defaults(fn=_reprocess)

    i = sub.add_parser("import",
                       help="copy a raw event log into a sharded run layout")
    i.add_argument("--raw", required=True,
                   help="raw_events.jsonl, or a run dir holding one")
    i.add_argument("--out", required=True, help="new run dir; must not exist")
    i.add_argument("--turns", metavar="PATH",
                   help="turns.jsonl to copy in; serve picks it up without "
                        "--turns afterwards")
    i.set_defaults(fn=_import)

    p = sub.add_parser("profile",
                       help="rebuild snapshots/trees from an imported run")
    p.add_argument("--run", required=True, help="run dir, or a raw event log")
    p.add_argument("--out", required=True,
                   help="output dir; must differ from --run")
    p.add_argument("--snapshot-interval", type=float, default=5.0)
    p.add_argument("--tree-dump-interval", type=float, default=15.0,
                   help="0 disables the periodic tree dumps")
    p.set_defaults(fn=_profile)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        # Bad paths and refusals to clobber a run are user errors, not crashes.
        print(f"kvtree {a.cmd}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
