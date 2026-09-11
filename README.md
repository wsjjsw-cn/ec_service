# 智能电商客服：两级语义缓存 + ReAct 检索中间件

> 面向 LLM 电商客服的「语义缓存 + 智能检索」中间件。在保证回答质量的前提下，缓存命中直接复用、语义近似补全复用、未命中才走完整 ReAct 检索，从而大幅降低大模型调用与 Token 消耗。

`Python` · `FastAPI` · `LangChain` · `LangGraph` · `bge-m3` · `Redis Stack` · `DeepSeek`

---

## 实测效果

本地 bge-m3 + DeepSeek 官方 API，30 条电商评测集：

| 指标 | 不使用缓存 | 使用缓存 | 优化 |
|---|---|---|---|
| LLM 调用次数 | 67 | 36 | **↓ 46%** |
| 总 Token | 52,911 | 29,870 | **↓ 43.5%** |
| 输入 Token | 43,747 | 16,952 | **↓ 61%** |
| 直接命中率（冷启动） | — | 48% | — |
| 缓存覆盖率（冷启动） | — | 56% | — |

> 对照口径：**不使用缓存** = 每条提问都走完整 RAG；**使用缓存** = 用户首次提问的冷启动轮。
> 延迟与 Token 均为真实墙钟计时 + 真实 Token 计量，基线由同一 pipeline 关闭缓存实跑得出，结果可复现。

---

## 核心亮点

- **两级混合缓存**：L1 规则缓存（归一化精确匹配、意图捷径、编辑距离、子问题重叠）毫秒级命中；L2 语义缓存基于本地 bge-m3 真实语义向量召回，覆盖「字面不同、语义相同」的问法。
- **动态查询前置拦截**：库存、价格、物流单号、预售发货时间等实时问题在进入缓存前被识别并拦截，杜绝用静态答案回答时效性问题。
- **Validator 三级裁决**：轻量 LLM 对候选答案判定「直接复用 / 检索补全后复用 / 回退完整 RAG」，在复用收益与回答准确性之间取得平衡。
- **ReAct 检索（LangGraph）**：缓存未命中时由 ReAct 智能体做工具化深度检索并生成答案。
- **缓存回写闭环**：未命中的问答结果按规则写回 L1/L2，系统持续积累可复用答案，形成正向闭环。
- **可复现的离线评测**：真实墙钟计时 + 真实 Token 计量，基线用同一 pipeline 关闭缓存实跑，指标可信、可复现。

---

## 请求链路

```text
用户提问
   │
   ▼
动态查询拦截 ──命中──▶ 实时数据直答（不进静态缓存）
   │ 未命中
   ▼
L1 规则缓存 ──命中──▶ 直接复用（毫秒级）
   │ 未命中
   ▼
L2 语义缓存（bge-m3 向量召回）
   │ 候选
   ▼
Validator 三级裁决 ──复用 / 补全──▶ 复用或补全生成 ──▶ 回写 L1/L2
   │ 回退
   ▼
ReAct 检索（LangGraph）→ 答案生成 ───────────────▶ 回写 L1/L2
```

---

## 核心设计

### 两级缓存
- **L1 规则缓存**：归一化精确匹配、意图捷径、编辑距离与子问题重叠，命中即返回，不产生 LLM 调用。
- **L2 语义缓存**：以本地 bge-m3 生成真实语义向量做相似度召回，解决「换个说法问同一件事」的命中问题。

### 动态查询拦截
库存、价格、物流单号、预售发货时间等实时问题绕过静态缓存，直接从结构化数据源作答，保证时效性正确。

### Validator 三级裁决
对 L2 给出的候选答案做三态判定：`REUSE`（直接复用）、`COMPLETE`（检索补全后复用）、`RAG`（缓存收益不足，回退完整检索），并据此驱动缓存回写。

### ReAct 检索与缓存回写
缓存完全未命中时，由 LangGraph 上的 ReAct 智能体发起工具化检索并生成答案；生成结果按规则写回 L1/L2，使系统持续沉淀可复用答案。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 服务 | FastAPI + Uvicorn |
| 编排 | LangChain + LangGraph（ReAct） |
| 向量 | 本地 bge-m3（GGUF / llama.cpp，CPU 可跑） |
| 缓存 | Redis Stack（RedisVL HNSW 向量索引）/ 内存后端 |
| 大模型 | DeepSeek（OpenAI 兼容接口） |

---

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
docker compose up -d redis-stack      # 可选，不启动则自动走内存缓存
python scripts/run_eval.py            # 离线评测
python scripts/run_api.py             # 启动服务，访问 http://127.0.0.1:8000/
```

---

## LLM 配置

在项目根目录创建 `.env`（参考 `.env.example`，该文件已被 `.gitignore` 排除）：

```ini
# DeepSeek 官方接口
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-flash
LLM_PROVIDER=deepseek

# 其它 OpenAI 兼容网关（如 SiliconFlow）：改 LLM_PROVIDER=openai 并填入对应 base_url / model
```

未配置密钥时系统自动降级为规则模式，评测报告会明确标注指标为规则降级模式下的值。

---

## Embedding 后端

语义缓存的效果取决于向量模型。系统按以下优先级自动选择，**优先使用本地模型，无需联网下载**：

| 优先级 | 后端 | 说明 |
|---|---|---|
| 1 | `llama-cpp` | 本地 GGUF 模型（默认 bge-m3，1024 维，CPU 可跑） |
| 2 | `fastembed` | ONNX 推理的 `BAAI/bge-small-zh-v1.5`，无需 torch |
| 3 | `sentence-transformers` | 效果最好，依赖 torch |
| 4 | `lsa` | 纯 numpy，在知识库语料上拟合，可捕捉共现语义 |
| 5 | `hashing` | 零依赖兜底，只能捕捉字面重叠 |

```powershell
$env:SEMANTIC_CS_EMBEDDING_BACKEND = "fastembed"          # 强制指定后端
$env:SEMANTIC_CS_LOCAL_EMBEDDING_MODEL = "D:\models\bge-m3-Q8_0.gguf"
```

> 采用真实语义向量后，同义改写的相似度可达 0.85~0.98，显著优于哈希 / 词袋方案。

---

## 阈值校准

阈值与模型分布强相关，不能主观设定。运行：

```powershell
python scripts/calibrate_thresholds.py
```

脚本用三组标注样本扫描阈值：FAQ 的同义改写（正例）、域外问题（负例）、以及**业务域内但 FAQ 无对应条目、本应走 RAG 的问题**（负例）——最后一组能有效避免缓存复用偏题答案。

bge-m3 上的标定结果：

| 参数 | 值 | 依据 |
|---|---|---|
| `l2_similarity_threshold` | 0.76 | 正例 min 0.757 / 域外 max 0.529；0.76 时召回 95.7%，F1 最优 |
| `validator_reuse_threshold` | 0.80 | 高精度区，直接复用 |
| `validator_complete_threshold` | 0.76 | 0.76~0.80 先检索补全再复用 |

---

## 离线评测

`scripts/run_eval.py` 输出，核心保证：

- **真实墙钟计时**，不使用任何硬编码延迟常量。
- **真实基线**：同一 pipeline 以 `bypass_cache=True` 跑完整 RAG 作为对照。
- **冷启动轮为主指标**：反映用户首次提问的真实命中能力。
- **幂等可复现**：独立 pipeline，每轮前重置缓存与 embedding 编码缓存。

```text
total=30                     dynamic_intercepts=5
cache_coverage_rate=56.00%   direct_cache_reuse_rate=48.00%   (cold)
LLM 调用   67 → 36            token 52,911 → 29,870（-43.5%）
```

---

## API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"七天无理由退货需要满足什么条件？\"}"
```

| 接口 | 说明 |
|---|---|
| `POST /chat` | 问答，返回 route / cache_level / decision / latency_ms / llm_calls |
| `GET /metrics/eval` | 离线评测指标（幂等） |
| `GET /health` | 健康检查 |

---

## 目录结构

```text
data/                  FAQ、知识库、商品与离线评测集
frontend/              客服演示控制台
scripts/               预热、评测、阈值校准入口
src/semantic_cs/       系统源码
tests/                 单元测试与回归测试
```
