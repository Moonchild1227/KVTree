"""Durable run layout shared by live collection and offline profiling."""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.parse
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, path)


def stream_slug(stream: str) -> str:
    return urllib.parse.quote(stream, safe="")


class RunWriter:
    """Append batches to hourly, per-stream shards and maintain a manifest."""

    def __init__(self, root: Path, metadata: dict | None = None):
        self.root = root
        if (root / "manifest.json").exists():
            raise FileExistsError(f"run already exists: {root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.count = 0
        self.streams: dict[str, dict] = {}
        self.files: dict[Path, object] = {}
        self.metadata = metadata or {}
        self.last_manifest = 0.0
        self._write_manifest("running")

    def append(self, record: dict, encoded: str | None = None) -> None:
        ts = float(record["recv_ts"])
        stream = record["stream"]
        hour = time.strftime("%Y-%m-%dT%H", time.gmtime(ts))
        path = self.root / "events" / f"stream={stream_slug(stream)}" / f"{hour}.jsonl"
        fp = self.files.get(path)
        if fp is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fp = self.files[path] = path.open("a", buffering=1)
        fp.write(encoded if encoded is not None else json.dumps(record) + "\n")
        stat = self.streams.setdefault(stream, {"batches": 0, "first_ts": ts,
                                                 "last_ts": ts, "shards": []})
        stat["batches"] += 1
        stat["last_ts"] = ts
        rel = str(path.relative_to(self.root))
        if rel not in stat["shards"]:
            stat["shards"].append(rel)
        self.count += 1
        if time.time() - self.last_manifest >= 5:
            self._write_manifest("running")

    def _write_manifest(self, status: str) -> None:
        atomic_json(self.root / "manifest.json", {
            "format": "kvtree-run", "version": 1, "status": status,
            "started_at": self.started, "updated_at": time.time(),
            "batches": self.count, "streams": self.streams,
            "metadata": self.metadata,
        })
        self.last_manifest = time.time()

    def close(self) -> None:
        for fp in self.files.values():
            fp.close()
        self.files.clear()
        self._write_manifest("complete")


def event_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    shards = sorted((source / "events").glob("stream=*/*.jsonl"))
    if shards:
        return shards
    legacy = source / "raw_events.jsonl"
    return [legacy] if legacy.exists() else []


def import_run(raw: Path, out: Path, turns: Path | None = None) -> dict:
    if (out / "manifest.json").exists():
        raise FileExistsError(f"run already exists: {out}")
    if not raw.exists():
        raise FileNotFoundError(f"input does not exist: {raw}")
    paths = event_files(raw)
    if not paths:
        raise FileNotFoundError(f"no event files found under: {raw}")
    writer = RunWriter(out, {"imported_from": str(raw)})
    bad = 0
    for path in paths:
        for line in path.open():
            try:
                writer.append(json.loads(line))
            except (ValueError, KeyError, TypeError):
                bad += 1
    writer.close()
    if turns:
        dest = out / "turns" / "turns.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(turns, dest)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["bad_lines"] = bad
    atomic_json(out / "manifest.json", manifest)
    return manifest
