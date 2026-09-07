"""kvtree command line entry point."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from . import __version__


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
    serve(Path(a.dir), a.turns, a.port, a.bind)
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
    print(f"[import] {result['batches']} batches -> {a.out}")
    return 0


def _profile(a) -> int:
    from .profile import build
    result = build(Path(a.run), Path(a.out or a.run), a.snapshot_interval,
                   a.tree_dump_interval)
    print(f"[profile] {result['events']} events -> {result['profile_dir']}")
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
    s.add_argument("--turns", default=None, help="turns.jsonl for the session panels")
    s.add_argument("--port", type=int, default=8899)
    s.add_argument("--bind", default="0.0.0.0")
    s.set_defaults(fn=_serve)

    g = sub.add_parser("metrics", help="sample the engine's /metrics pool gauges")
    g.add_argument("--url", default="http://127.0.0.1:37000/metrics")
    g.add_argument("--out", required=True)
    g.add_argument("--interval", type=float, default=5.0)
    g.add_argument("--duration", type=float, default=0.0)
    g.set_defaults(fn=_metrics)

    r = sub.add_parser("reprocess",
                       help="rebuild snapshots/trees from recorded raw events")
    r.add_argument("--raw", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--snapshot-interval", type=float, default=5.0)
    r.add_argument("--tree-dump-interval", type=float, default=15.0)
    r.set_defaults(fn=_reprocess)

    i = sub.add_parser("import", help="import raw event logs into a run layout")
    i.add_argument("--raw", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--turns")
    i.set_defaults(fn=_import)

    p = sub.add_parser("profile", help="profile an imported or recorded run")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--snapshot-interval", type=float, default=5.0)
    p.add_argument("--tree-dump-interval", type=float, default=15.0)
    p.set_defaults(fn=_profile)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
