# kvtree

实时观测 SGLang 的 KV radix tree：订阅 kv-events 事件流，重建每个 attn-dp rank
的前缀树，把缓存分层、树形状、session 时间线放在同一条时间轴上。

## 安装

```bash
pip install -e .
```

只依赖 `pyzmq` 和 `msgspec`。前端是零构建的原生 JS。

## 引擎侧准备

启动 SGLang 时打开 kv-events，`--page-size` 要跟你的路由/客户端一致：

```bash
python -m sglang.launch_server ... \
  --page-size 16 \
  --enable-metrics --enable-metrics-for-all-schedulers --enable-cache-report \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:37004"}'
```

每个 attn-dp rank 一个 publisher，端口是 `base_port + dp_rank`。
`--enable-cache-report` 决定 usage 里有没有 `cached_tokens`（session 面板按前缀
命中率上色要用）。

## 用法

```bash
# 采集：订阅 4 个 rank
kvtree monitor --hosts 127.0.0.1 --base-port 37004 --dp-size 4 --out-dir ./data/kvmon

# 采样引擎池子水位（L1 Pool 面板需要）
kvtree metrics --url http://127.0.0.1:37000/metrics --out ./data

# 看板
kvtree serve --dir ./data/kvmon --turns ./data/turns.jsonl --port 8899
```

打开 `http://<host>:8899/`。

多机部署把 `--hosts` 写成逗号分隔：`--hosts 10.0.0.1,10.0.0.2 --dp-size 4` 会订阅
2×4 = 8 个 publisher。

### 离线重算

`raw_events.jsonl` 保存了每条解码后的事件，改了推导规则不用重跑引擎：

```bash
kvtree reprocess --raw ./data/kvmon/raw_events.jsonl --out ./data/kvmon_v2
kvtree serve --dir ./data/kvmon_v2 --turns ./data/turns.jsonl
```

### 命令与参数

```
kvtree monitor    --hosts --base-port --dp-size --topic --out-dir
                  --snapshot-interval --tree-dump-interval --duration
                  --sub-hwm --strict-schema --kv-events-py
kvtree metrics    --url --out --interval --duration
kvtree serve      --dir --turns --port --bind
kvtree reprocess  --raw --out --snapshot-interval --tree-dump-interval
```

`--strict-schema` 用引擎自带的 `kv_events.py` 解码；默认用内置的宽松 schema，
因为部分引擎发出的 `token_ids` 是 `(tok, next)` 成对结构，与其自身声明的
`list[int]` 不符，用原版 schema 会解码失败。

## 看板

- **KV Cache** — 分层驻留的 token 数与 block 数（L1 GPU / L2 CPU_PINNED /
  L3 EXTERNAL）。`write_through` 下一个 block 会同时驻留多层，归属取最快的那层。
- **L1 Pool** — 引擎的 `num_used_tokens`（被运行中请求引用）、
  `kv_evictable_tokens`（驻留但未被引用）、`kv_available_tokens`，跟事件流重建
  出的驻留量画在一起互相校验。三者常差两三个数量级，用工具栏的**对数刻度**看；
  `free` 默认隐藏，点图例可显示。
- **Radix Tree** — roots / leaves / 深度 / 主干长度随时间变化，加任意时刻的
  node-link 树图（长链折叠成 `×N`，滚轮缩放、拖拽平移、双击复位）。
  树图一次显示**一个 rank**，用 `stream` 下拉框切换。
- **Sessions** — 每行一个 session，每轮一个色块（按前缀命中率上色），块之间的
  间隙是等 tool 的时间；另有 in-flight 请求数与等 tool 的 session 数。

工具栏：
- 时间范围支持 `now` / `now-15m` 这类表达式，每次刷新重新求值，默认
  `全部 → now`；框选可缩放到绝对区间，`↺` 回到默认
- **Group By** 聚合 / 实例：聚合按层级堆叠；分实例时改为叠加折线，色相表示层级、
  明度表示实例
- 刷新间隔 Off / 5s / 10s / 30s

## 数据布局

```
data/
  kvmon/
    raw_events.jsonl      每个事件批次一行，可重放的事实来源
    snapshots.jsonl       周期快照：分层统计、树统计、序号连续性
    trees/tree_<ts>.json  周期性完整树快照
    live_tree.json        最新一份
  metrics.jsonl           引擎 /metrics 采样
  turns.jsonl             压测客户端写的每轮记录
```

`turns.jsonl` 每行需要这些字段，任何压测器按格式追加写即可（参考
`examples/replay_agentic.py`）：

```json
{"session_id": "s1", "turn": 0, "sent_at_ns": 0, "recv_at_ns": 0,
 "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
 "client_e2e_ms": 0, "tool_sleep_ms": 0, "http_status": 200,
 "finish_reason": "stop"}
```

## 限制

- 只支持 SGLang。
- 只能看到采集启动之后的事件，之前已缓存的 block 不在树里。要干净的树就在压测前
  重启引擎。
- `raw_events.jsonl` 会持续增长，目前需要手动轮转。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

## 许可

Apache-2.0
