"""kvtree — live KV radix tree observability for SGLang.

Subscribes to SGLang's kv-events stream, reconstructs the per-rank radix tree,
and serves a dashboard that puts cache tiers, tree shape and per-session
timelines on one shared time axis.
"""

__version__ = "0.1.0"

__all__ = ["collect", "metrics", "reprocess", "server", "__version__"]
