#!/usr/bin/env python3
"""Reference load generator: replay recorded agent rollouts against SGLang.

This is an EXAMPLE, not part of the kvtree package. It expects a parquet with
one row per rollout and these columns:

  task_type                    filter value, e.g. "swe"
  rollout_id                   "<data_id>:<gen_id>"; rows sharing data_id form a group
  rle_session_id               session identity
  agentic_final_traj_messages  zlib+base64 JSON chat messages
  turn_records                 JSON [{turn_idx, token_span, tool_elapsed_ms}, ...]

Adapt load_sessions() to your own dataset. The only thing kvtree cares about is
the turns.jsonl this writes -- see README.

The parquet holds finished agent trajectories:
  agentic_final_traj_messages : zlib+base64 JSON chat messages (system/user/
                                assistant/tool/...)
  turn_records                : [{turn_idx, token_span, tool_elapsed_ms}, ...]

One rollout = one session. Turn k replays the conversation prefix up to the
k-th assistant message and asks the engine to regenerate it; between turns we
sleep the recorded tool_elapsed_ms, which is what makes the load agentic (idle
gaps during which the session's KV blocks sit in the radix tree unused).

Writes turns.jsonl in the shape kvtree's session panels expect, so the
dashboard shows sessions and the KV tree on one shared axis.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import zlib
from pathlib import Path

import httpx
import pandas as pd


def load_sessions(uri: str, groups: int, rollout_n: int) -> list[dict]:
    df = pd.read_parquet(uri)
    total = len(df)
    df = df[df["task_type"] == "swe"]
    # rollout_id = "<data_id>:<gen_id>"; a group is one data_id
    data_id = df["rollout_id"].str.rsplit(":", n=1).str[0]
    keep = list(dict.fromkeys(data_id))[:groups]
    # loud counters: a silent 0-session load once cost a full engine restart
    print(f"[replay] dataset {uri}: {total} rows, {len(df)} swe, "
          f"{data_id.nunique()} groups -> using {len(keep)}", flush=True)
    out: list[dict] = []
    for gid in keep:
        rows = df[data_id == gid].head(rollout_n)
        for pos, (_, r) in enumerate(rows.iterrows()):
            msgs = json.loads(
                zlib.decompress(base64.b64decode(r["agentic_final_traj_messages"]))
            )
            recs = json.loads(r["turn_records"])
            asst = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            if not asst:
                continue
            out.append({
                "session_id": f"{r['rle_session_id']}-{pos}",
                "messages": msgs,
                "assistant_idx": asst,
                "tool_ms": [float(t.get("tool_elapsed_ms") or 0.0) for t in recs],
            })
    return out


def _content_len(msg: dict) -> int:
    c = msg.get("content")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(str(p.get("text", ""))) for p in c if isinstance(p, dict))
    return 0


async def run_session(
    client: httpx.AsyncClient,
    base: str,
    model: str,
    s: dict,
    sink: list,
    max_turns: int,
    tool_scale: float,
    fp=None,
) -> None:
    for turn, idx in enumerate(s["assistant_idx"][:max_turns]):
        prefix = s["messages"][:idx]
        target = s["messages"][idx]
        # ask for roughly what the agent actually produced, capped for safety
        want = max(16, min(1024, _content_len(target) // 3 + 16))
        body = {
            "model": model,
            "messages": prefix,
            "max_tokens": want,
            "temperature": 0.0,
            "stream": False,
        }
        sent = time.time()
        status, usage, finish = 0, {}, ""
        try:
            r = await client.post(f"{base}/v1/chat/completions", json=body,
                                  timeout=600.0)
            status = r.status_code
            if status == 200:
                p = r.json()
                usage = p.get("usage") or {}
                finish = (p.get("choices") or [{}])[0].get("finish_reason") or ""
        except Exception as exc:  # network / timeout: record and keep going
            finish = f"{type(exc).__name__}"
        recv = time.time()
        tool_ms = (s["tool_ms"][turn] if turn < len(s["tool_ms"]) else 0.0) * tool_scale
        det = usage.get("prompt_tokens_details") or {}
        rec = {
            "session_id": s["session_id"],
            "turn": turn,
            "sent_at_ns": int(sent * 1e9),
            "recv_at_ns": int(recv * 1e9),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": det.get("cached_tokens", 0),
            "client_e2e_ms": (recv - sent) * 1000.0,
            "meta_queue_time_s": 0.0,
            "tool_sleep_ms": tool_ms,
            "meta_dp_rank": -1,
            "http_status": status,
            "finish_reason": finish,
            "is_warmup": False,
        }
        sink.append(rec)
        if fp is not None:      # stream it out: the live dashboard should show
            fp.write(json.dumps(rec) + "\n")   # each turn as it lands
        if tool_ms > 0:
            await asyncio.sleep(tool_ms / 1000.0)


async def main_async(a) -> int:
    sessions = load_sessions(a.uri, a.groups, a.rollout_n)
    planned = sum(len(s["assistant_idx"][:a.max_turns]) for s in sessions)
    print(f"[replay] {len(sessions)} sessions, {planned} turns planned")
    if not sessions or not planned:
        print("[replay] ABORT: nothing to replay -- refusing to report success")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    turns_path = out / "turns.jsonl"
    fp = turns_path.open("a", buffering=1)
    sink: list = []
    sem = asyncio.Semaphore(a.concurrency)
    limits = httpx.Limits(max_connections=a.concurrency + 8)

    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        try:
            info = (await client.get(f"{a.base}/get_model_info", timeout=30.0)).json()
            model = info.get("model_path") or a.model
        except Exception:
            model = a.model
        print(f"[replay] model={model}")

        async def one(s):
            async with sem:
                await run_session(client, a.base, model, s, sink,
                                  a.max_turns, a.tool_scale, fp)

        t0 = time.time()
        await asyncio.gather(*(one(s) for s in sessions))
        dur = time.time() - t0

    fp.close()
    ok = sum(1 for r in sink if r["http_status"] == 200)
    pt = sum(r["prompt_tokens"] for r in sink)
    ct = sum(r["cached_tokens"] for r in sink)
    print(f"[replay] done in {dur:.1f}s: {ok}/{len(sink)} ok, "
          f"prefix hit {100.0 * ct / max(1, pt):.1f}%  -> {turns_path}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:37000")
    ap.add_argument("--uri", required=True,
                    help="parquet of recorded rollouts (see module docstring)")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--rollout-n", type=int, default=16, dest="rollout_n")
    ap.add_argument("--max-turns", type=int, default=8, dest="max_turns")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tool-scale", type=float, default=1.0, dest="tool_scale",
                    help="scale recorded tool wait (0.25 = 4x faster replay)")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True, help="dir for turns.jsonl")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
