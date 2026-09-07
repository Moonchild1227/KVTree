"""HTTP API + static dashboard for a kvtree data directory."""

from __future__ import annotations

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

STATIC = Path(__file__).resolve().parent / "static"


def make_handler(root: Path, turns_path: str | None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _snapshots(self):
            f = root / "snapshots.jsonl"
            out = []
            if f.exists():
                for line in f.read_text().splitlines()[-4000:]:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    # keep per-stream tree stats / counters: the shape and
                    # event-rate panels are derived from them
                    streams = {}
                    for k, v in (r.get("streams") or {}).items():
                        streams[k] = {
                            "tree": v.get("tree") or {},
                            # per-instance tier split: the dashboard's
                            # "Group By: instance" needs it, aggregate totals
                            # alone hide which rank is hot
                            "mediums": v.get("mediums") or {},
                            "stored": v.get("stored", 0),
                            "removed": v.get("removed", 0),
                            "batches": v.get("batches", 0),
                            "seq_gaps": v.get("seq_gaps", 0),
                        }
                    out.append({"ts": r["ts"], "time": r["time"],
                                "totals": r["totals"], "streams": streams})
            return out

        def _trees(self):
            import time as _t
            out = []
            d = root / "trees"
            if d.exists():
                for p in sorted(d.glob("tree_*.json")):
                    m = re.match(r"tree_(\d+)\.json", p.name)
                    if m:
                        out.append({"ts": int(m.group(1))})
            live = root / "live_tree.json"
            if live.exists():
                try:
                    lt = json.loads(live.read_text())
                    if not out or out[-1]["ts"] != int(lt["ts"]):
                        out.append({"ts": int(lt["ts"]), "live": True})
                except Exception:
                    pass
            for it in out:
                it["time"] = _t.strftime("%H:%M:%S", _t.localtime(it["ts"]))
            return out

        def _turns(self):
            out = []
            p = Path(turns_path) if turns_path else root / "turns" / "turns.jsonl"
            if not p.exists():
                return out
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                    if r.get("is_warmup"):
                        continue
                    out.append({
                        "session_id": r["session_id"],
                        "turn": r["turn"],
                        "sent": r["sent_at_ns"] / 1e9,
                        "recv": (r["recv_at_ns"] or r["sent_at_ns"]) / 1e9,
                        "prompt_tokens": r.get("prompt_tokens", 0),
                        "completion_tokens": r.get("completion_tokens", 0),
                        "cached_tokens": r.get("cached_tokens", 0),
                        "e2e_ms": r.get("client_e2e_ms", 0),
                        "queue_s": r.get("meta_queue_time_s", 0),
                        "tool_sleep_ms": r.get("tool_sleep_ms", 0),
                        "dp_rank": r.get("meta_dp_rank"),
                        "http_status": r.get("http_status", 0),
                        "finish_reason": r.get("finish_reason", ""),
                    })
                except Exception:
                    pass
            return out

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send((STATIC / "index.html").read_bytes(),
                           "text/html; charset=utf-8")
            elif u.path.startswith("/static/"):
                name = u.path[len("/static/"):]
                f = (STATIC / name).resolve()
                if STATIC not in f.parents or not f.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                self._send(f.read_bytes(), ctype)
            elif u.path == "/api/snapshots":
                self._send(json.dumps(self._snapshots()).encode())
            elif u.path == "/api/trees":
                self._send(json.dumps(self._trees()).encode())
            elif u.path == "/api/metrics":
                out = []
                mp = root.parent / "metrics.jsonl"
                if not mp.exists():
                    mp = root / "metrics.jsonl"
                if mp.exists():
                    for line in mp.read_text().splitlines()[-4000:]:
                        try:
                            r = json.loads(line)
                            out.append({"ts": r["ts"], "time": r["time"],
                                        "total": r.get("total") or {},
                                        "ranks": r.get("ranks") or {}})
                        except Exception:
                            pass
                self._send(json.dumps(out).encode())
            elif u.path == "/api/turns":
                self._send(json.dumps(self._turns()).encode())
            elif u.path == "/api/tree":
                ts = parse_qs(u.query).get("ts", [""])[0]
                p = root / "trees" / f"tree_{ts}.json"
                if not p.exists():
                    try:
                        p = root / "trees" / f"tree_{int(float(ts) * 1000)}.json"
                    except (TypeError, ValueError):
                        pass
                if not p.exists():
                    p = root / "live_tree.json"
                self._send(p.read_bytes() if p.exists()
                           else b'{"ts":0,"streams":{}}')
            else:
                self.send_response(404)
                self.end_headers()

    return H


def make_server(directory: Path, turns: str | None, port: int = 8899,
                bind: str = "0.0.0.0") -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((bind, port), make_handler(directory, turns))
    used = Path(turns) if turns else directory / "turns" / "turns.jsonl"
    print(f"kvtree dashboard: http://{bind}:{port}/  (data: {directory}"
          + (f", turns: {used}" if used.exists() else "") + ")", flush=True)
    return srv


def serve(directory: Path, turns: str | None, port: int = 8899,
          bind: str = "0.0.0.0") -> None:
    make_server(directory, turns, port, bind).serve_forever()
