# kvtree

在 agentic 负载下实时观测 SGLang 的 **KV radix tree**。

`/metrics` 只能给你标量水位（用了多少、还剩多少）。kvtree 订阅 SGLang 的
kv-events 事件流，把**树的拓扑本身**重建出来——哪些前缀被共享、每个 block
驻留在哪一层、同一段 KV 在几个 dp rank 里重复了几份、等 tool 的空隙里有多少
缓存在白占着。这些是水位指标回答不了的问题。

## 它回答什么

- **分层驻留**：L1(GPU) / L2(CPU_PINNED) / L3(EXTERNAL) 各有多少 block 和 token。
  `write_through` 下一个 block 会同时驻留 GPU 和 CPU，kvtree 按**最快的那一层**
  归属（GPU 优先），而不是被后写的事件覆盖。
- **树形状**：roots / leaves / 最大深度 / 主干长度 / 分叉节点数随时间的变化，
  以及任意时刻的 node-link 树图（长链自动折叠成 `×N`）。
- **闲置缓存**：把引擎的 `num_used_tokens`（被运行中请求引用）和
  `kv_evictable_tokens`（驻留但没被引用）跟树的驻留量画在同一张图上。
  实测某次 SWE 回放里，引用中 173k token、闲置驻留 1946k——**闲置是在用的 11 倍**。
- **跨 rank 重复**：block hash 是内容寻址的，所以可以直接统计同一段 KV 在几个
  attn-dp rank 里各存了一份。实测无调度基线下 **62.7% 的 L1 存储是跨 rank 重复**，
  这是 KV 感知调度可以回收的空间。
- **session 时间线**：每行一个 session，每轮对话一个色块（按前缀命中率上色），
  块之间的间隙就是等 tool 的时间。所有以时间为横轴的面板共享同一根游标。

## 安装

```bash
pip install -e .
```

只依赖 `pyzmq` 和 `msgspec`。前端是零构建的原生 JS，没有 node 工具链。

## 用法

引擎侧需要打开 kv-events（每个 attn-dp rank 一个 publisher，端口
`base_port + dp_rank`），并保证 `--page-size` 与之匹配：

```bash
python -m sglang.launch_server ... \
  --page-size 16 \
  --enable-metrics --enable-metrics-for-all-schedulers --enable-cache-report \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:37004"}'
```

然后：

```bash
# 1. 采集：订阅 4 个 rank，落盘原始事件 + 快照 + 树 dump
kvtree monitor --hosts 127.0.0.1 --base-port 37004 --dp-size 4 --out-dir ./data/kvmon

# 2. 采样引擎池子水位（可选，但"闲置缓存"那张图需要它）
kvtree metrics --url http://127.0.0.1:37000/metrics --out ./data

# 3. 看板
kvtree serve --dir ./data/kvmon --turns ./data/turns.jsonl --port 8899
```

打开 `http://<host>:8899/`。时间范围默认 `全部 → now`，跟 Grafana 一样支持
`now-15m` / `now` 这类表达式，配合右上角的刷新间隔持续更新。

### 离线重放

`raw_events.jsonl` 保存了每一条解码后的事件，所以**推导规则变了不用重跑引擎**：

```bash
kvtree reprocess --raw ./data/kvmon/raw_events.jsonl --out ./data/kvmon_v2
```

分层归属规则改过一次之后，15328 条事件几秒钟就重算完，结论直接从
"全是 L2" 翻转成 "全是 L1 双驻留"。这条路径是设计目标，不是应急手段。

## 数据布局

```
data/
  kvmon/
    raw_events.jsonl    每个事件批次一行（可重放的事实来源）
    snapshots.jsonl     每 5s 一行：分层统计 + 树统计 + 序号连续性
    trees/tree_<ts>.json  周期性的完整树快照
    live_tree.json      最新一份
  metrics.jsonl         引擎 /metrics 采样
  turns.jsonl           压测客户端写的每轮记录（session 面板的输入）
```

`turns.jsonl` 每行需要的字段见 `kvtree/server.py` 的 `_turns()`：
`session_id / turn / sent_at_ns / recv_at_ns / prompt_tokens /
completion_tokens / cached_tokens / client_e2e_ms / tool_sleep_ms /
http_status / finish_reason`。任何压测器只要按这个格式追加写，泳道就能用；
`examples/replay_agentic.py` 是一个参考实现。

## 已知限制

- **只支持 SGLang**。vLLM 的 kv-events 协议接近，但没做适配。
- 事件流反映的是**引擎发出的事件**，采集启动前就已缓存的 block 看不到，
  表现为"命中但树上没有"。要干净的树就在压测前重启引擎。
- `raw_events.jsonl` 会持续增长（实测约 12 分钟 2.7MB）。目前需要手动轮转。
- 树图一次只显示**一个 rank**（下拉框切换）；曲线面板是跨 rank 聚合的，
  两者的数字对不上是正常的。

## 踩过的坑（都有回归测试）

- DSv4 发出的 `token_ids` 可能是 `(tok, next)` 成对结构，而引擎自己的
  `kv_events.py` 声明是 `list[int]`——**引擎解不开自己发的数据**。kvtree 默认用
  更宽松的镜像 schema，`--strict-schema` 可切回引擎原版。
- `write_through` 会为同一个 block 连发 GPU 和 CPU 两条 `BlockStored`，
  按"后写覆盖"会把整棵树标成 L2。
- `BlockRemoved` 带 medium：从 GPU 淘汰不等于节点消失，它可能仍在 L2。

## 许可

Apache-2.0
