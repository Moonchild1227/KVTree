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

`observe` 一条命令把三件事拉起（monitor + metrics + dashboard，一个 Ctrl-C 全停）。
kvtree 不负责启动 SGLang——先起 observe（或 monitor），再起引擎和 workload，事件才不会漏：

```bash
kvtree observe \
  --hosts 10.99.90.190,10.99.90.187,10.99.90.199,10.99.90.196 \
  --base-port 37004 --dp-size 4 \
  --metrics-url http://127.0.0.1:37000/metrics \
  --out-dir ./data/kvmon \
  --turns ./client/turns.jsonl \
  --port 8899
```

session 文件不自动发现：压测客户端把它写在哪里，kvtree 就从哪里接。显式 `--turns`
优先；也可以导出通用环境变量 `KVTREE_TURNS=/path/to/client/turns.jsonl`，
之后 `observe` / `serve` 不传 `--turns` 也能找到。

也可以分开跑各组件：

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

采集下来的事件是可重放的事实来源，改了推导规则不用重跑引擎：

```bash
kvtree reprocess --raw ./data/kvmon/raw_events.jsonl --out ./data/kvmon_v2
kvtree serve --dir ./data/kvmon_v2 --turns ./data/turns.jsonl
```

长跑或者要归档的场景，先 `import` 成分片布局（按 stream + 小时切分，带
`manifest.json`），再 `profile`：

```bash
# 把裸日志（或者一个已有 run 目录）整理成 run 布局，顺手把 turns 收进去
kvtree import --raw ./data/kvmon/raw_events.jsonl --out ./runs/exp1 \
              --turns ./data/turns.jsonl

# 重算快照和树；--out 必须和 --run 不同，profile 不会就地覆盖录制结果
kvtree profile --run ./runs/exp1 --out ./runs/exp1_v2 --snapshot-interval 5

# turns 已经在 run 里，serve 不用再指
kvtree serve --dir ./runs/exp1_v2
```

`import` 和 `profile` 都拒绝写入已存在的 run / profile 目录——重复导入会把每条记录
写两遍，重复 profile 会覆盖已有结果。`reprocess` 是 `profile` 的旧前端，两者共用同一
套重放实现，区别只是 `reprocess` 允许反复写同一个输出目录。

### 命令与参数

```
kvtree observe    --hosts --base-port --dp-size --topic --out-dir
                  --snapshot-interval --tree-dump-interval --schema --sub-hwm
                  --metrics-url --metrics-interval --turns --port --bind
kvtree monitor    --hosts --base-port --dp-size --topic --out-dir
                  --snapshot-interval --tree-dump-interval --duration
                  --sub-hwm --schema
kvtree metrics    --url --out --interval --duration
kvtree serve      --dir --turns --port --bind
kvtree reprocess  --raw --out --snapshot-interval --tree-dump-interval
kvtree import     --raw --out --turns
kvtree profile    --run --out --snapshot-interval --tree-dump-interval
```

`observe` / `serve` 的 `--turns` 缺省时读环境变量 `KVTREE_TURNS`，再缺省才回落到
run 目录里的 `turns/turns.jsonl`。

`--tree-dump-interval 0` 关闭周期性树快照（`monitor` 和 `profile` 都是这个语义）。
`monitor` 可以反复写同一个 `--out-dir`，续采会接着写同一个 run。

`--schema /path/to/kv_events.py` 严格使用指定的引擎 schema 解码；默认使用内置的
宽松 schema，因为部分引擎发出的 `token_ids` 是 `(tok, next)` 成对结构，与其自身
声明的 `list[int]` 不符，用原版 schema 会解码失败。

## 看板

- **KV Cache** — 当前仍驻留的 KV token 数与 block 数（L1 GPU / L2 CPU_PINNED /
  L3 EXTERNAL），包含 used 和 idle，不是累计写入量。`write_through` 下一个 block
  会同时驻留多层，归属取最快的那层。**L3 是推断值**：引擎只在 GPU/CPU 两层发
  事件，进过 host 缓存的 block（即已排队 backup 到 mooncake）在最后一层驻留
  被驱逐后不删除、保留为推断的 EXTERNAL 块；mooncake 侧的驱逐不可见，所以
  长时间运行下 L3 只会偏多不会偏少。L3 的真实读写量看 L3 Mooncake 面板的
  prefetch/backup 计数器。
- **L1 Pool** — 引擎的 `num_used_tokens`（被运行中请求引用）、
  `kv_evictable_tokens`（驻留但未被引用）、`kv_available_tokens`，跟事件流重建
  出的驻留量画在一起互相校验。三者常差两三个数量级，用工具栏的**对数刻度**看；
  `free` 默认隐藏，点图例可显示。
- **L3 Mooncake / L2 Host Pool** — 引擎计数器：L3 prefetch（命中）与
  backup（写入）速率，L2 host 池水位。
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
    manifest.json         run 元信息：状态、批次数、每个 stream 的分片与时间范围
    raw_events.jsonl      每个事件批次一行，可重放的事实来源
    events/               同样的内容，按 stream + 小时分片
      stream=<url编码的stream>/<YYYY-MM-DDTHH>.jsonl
    snapshots.jsonl       周期快照：分层统计、树统计、序号连续性
    trees/tree_<ts>.json  周期性完整树快照
    live_tree.json        最新一份
    turns/turns.jsonl     `import --turns` 收进来的那份（serve 会自动找到）
  metrics.jsonl           引擎 /metrics 采样
  turns.jsonl             压测客户端写的每轮记录
```

`raw_events.jsonl` 和 `events/` 内容相同：前者是 `reprocess --raw` 的输入、也是分片
布局之前的格式，后者是 `profile` 长跑时读的。两份由同一次编码写出，等到没有东西再指
向扁平文件就可以删掉它。

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
- 事件日志会持续增长，目前需要手动轮转；而且同一份内容落两处
  （`raw_events.jsonl` 和 `events/`），磁盘按两倍算。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

## 许可

Apache-2.0
