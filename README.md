# 两级缓存与语义增强的智能电商客服

工程化复现项目，覆盖「动态拦截 → 两级缓存 → Validator 裁决 → ReAct 检索 → 缓存回写」全链路。

- 动态查询前置拦截：库存、价格、物流单号、预售发货时间等实时问题不进静态缓存
- L1 规则缓存：归一化精确匹配、意图捷径、编辑距离、子问题重叠
- L2 语义缓存：**真实语义向量**召回（本地 bge-m3），覆盖字面不同但语义相同的问法
- Validator 三级裁决：直接复用 / 检索补全后复用 / 回退完整 RAG
- ReAct 检索（LangGraph）与答案生成
- 缓存回写闭环：未命中的答案按规则写回 L1/L2，形成正向积累
- 离线评测：**真实墙钟计时 + 真实基线对比**，并输出指标可信度说明

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
docker compose up -d redis-stack      # 可选，不启动则自动走内存缓存
python scripts/run_eval.py            # 离线评测
python scripts/run_api.py             # 启动服务
```

访问 `http://127.0.0.1:8000/` 查看控制台。未连接后端时前端指标显示为 `--`，不会展示任何未经实测的数字。

## LLM 配置

在项目根目录创建 `.env`（可参考 `.env.example`，该文件已被 `.gitignore` 排除）：

```ini
# LLM：SiliconFlow 的 OpenAI 兼容接口（当前默认模型 MiniMaxAI/MiniMax-M2.5）
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=MiniMaxAI/MiniMax-M2.5
LLM_PROVIDER=openai

# DeepSeek 直连（备用：把 LLM_PROVIDER 改成 deepseek 即可切换）
# DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
# DEEPSEEK_MODEL=deepseek-v4-flash
```

未配置时系统自动降级为规则模式，评测报告会用 `notes` 明确标注指标不可信。

> **MiniMax-M2.5 是推理模型**：单次调用的输出 token 里绝大部分是 `reasoning`
> （实测占比可超 90%）。两点影响：`max_tokens` 必须留足余量（默认 2048），否则
> 推理吃满配额后 `content` 会是空字符串（代码已做空响应降级）；
> 成本必须按 token 计量，只统计调用次数会严重偏差。
> 价格按 SiliconFlow 官方定价：输入 $0.3/M、输出 $1.2/M（命中缓存输入 $0.03/M）。

## Embedding 后端

语义缓存的效果完全取决于向量模型。系统按以下优先级自动选择，**优先使用本地模型，无需联网下载**：

| 优先级 | 后端 | 说明 |
|---|---|---|
| 1 | `llama-cpp` | 本地 GGUF 模型，默认 `D:\vs_project\artical_knowledge\models\bge-m3-Q8_0.gguf`（1024 维，CPU 可跑） |
| 2 | `fastembed` | ONNX 推理的 `BAAI/bge-small-zh-v1.5`，无需 torch |
| 3 | `sentence-transformers` | 效果最好，依赖 torch |
| 4 | `lsa` | 纯 numpy，在知识库语料上拟合，可捕捉共现语义 |
| 5 | `hashing` | 零依赖兜底，只能捕捉字面重叠 |

可用环境变量覆盖：

```powershell
$env:SEMANTIC_CS_EMBEDDING_BACKEND = "fastembed"          # 强制指定后端
$env:SEMANTIC_CS_LOCAL_EMBEDDING_MODEL = "D:\models\bge-m3-Q8_0.gguf"
```

> 早期版本使用 md5 哈希词向量作为「语义」向量：同义改写的相似度会塌到 0.2~0.6，
> 导致 L2 缓存形同虚设。现默认使用本地 bge-m3，同义改写相似度提升到 0.85~0.98。

## 阈值校准

阈值不能拍脑袋定，必须与模型分布匹配。运行：

```powershell
python scripts/calibrate_thresholds.py
```

脚本用三组标注样本扫描阈值：FAQ 的同义改写（正例）、域外问题（负例）、
**业务域内但 FAQ 无对应条目、本应走 RAG 的问题**（负例）。最后一组很关键——
这类问题与 FAQ 的相似度分布（0.62~0.83）和同义改写高度重叠，漏掉它会让缓存
大量复用偏题答案。

当前在 bge-m3 上的标定结果：

| 参数 | 值 | 依据 |
|---|---|---|
| `l2_similarity_threshold` | 0.76 | 正例 min 0.757 / 域外 max 0.529 / 应走RAG 均值 0.719；0.76 时召回 95.7%、误复用 3/11，F1 最优 |
| `validator_reuse_threshold` | 0.80 | 高精度区，直接复用 |
| `validator_complete_threshold` | 0.76 | 0.76~0.80 先检索补全再复用 |

## 离线评测

`scripts/run_eval.py` 输出。评测要点：

- **真实墙钟计时**，不再使用硬编码的延迟常量
- **真实基线**：同一 pipeline 以 `bypass_cache=True` 跑完整 RAG 作为对照
- **冷启动轮为主指标**：用户首次提问的真实命中能力；稳态轮（缓存已含完全相同的问句）
  单列，命中率高是必然结果，不反映泛化能力
- **幂等**：使用独立 pipeline，每轮前重置缓存与 embedding 编码缓存
- **可信度标注**：未配置 `LLM_API_KEY`（LLM 未接入）时报告会明确说明指标为规则降级模式下的值

> 以下实测来自切换前的 **deepseek-v4-flash** 主跑（已完整跑完）。当前默认模型已切到
> **SiliconFlow MiniMax-M2.5**（单价已按 $0.3/$1.2 更新），但因完整 re-benchmark 被中途停止，
> 暂无 MiniMax 的端到端实测数字。结构与结论方向（token 省、成本被长 output 抵消）仍然成立。

当前环境（本地 bge-m3 + deepseek-v4-flash，30 条评测集）实测：

```
# 对照口径：左 = 不使用缓存(bypass_cache，每条都走完整 RAG)  右 = 使用缓存（冷启动轮）
total=30                     dynamic_intercepts=5
cache_coverage_rate=56.00%   direct_cache_reuse_rate=48.00%   (cold)
LLM 调用   67 → 36            token 52,911 → 29,870（-43.5%）
成本       $0.039796 → $0.036573（-8.1%）
延迟       3202ms → 3254ms（基本持平）
```

**这份数据比「成本省 32%、吞吐 +36%」难看，但它是真的**，而且暴露了两个真问题：

1. **token 省了 40%，成本只省 5.25%**：input token 从 41,086 降到 17,110（-58%），
   但 output token 从 8,726 涨到 12,841（+47%）——补全复用路径要生成「已有答案 + 补充说明」，
   比直接 RAG 的答案更长。output 单价是 input 的 4 倍，于是收益被吃掉大半。
2. **延迟基本没变**：LLM 调用耗时（约 2.9s）占绝对主导，调用次数少了 45%，
   但 COMPLETE 路径的 prompt 更长、生成更长，单次调用变慢，净收益被抵消。

对应的优化方向：精简 COMPLETE 的 prompt 与输出长度、提高直接复用比例、
Validator 换用非推理模型（推理模型单次调用约 90+ reasoning tokens，MiniMax-M2.5 实测占比超 90%）。

## API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"七天无理由退货需要满足什么条件？\"}"
```

| 接口 | 说明 |
|---|---|
| `POST /chat` | 问答，返回 route / cache_level / decision / latency_ms / llm_calls |
| `GET /metrics/eval` | 离线评测指标（幂等，含 notes 说明可信度前提） |
| `GET /health` | 健康检查 |

## 目录

```text
data/                  FAQ、知识库、商品和离线评测集
frontend/              客服演示控制台
scripts/               预热、评测、阈值校准入口
src/semantic_cs/        系统源码
tests/                 核心单元测试与回归测试
```

## 已知限制

- **成本收益被 output token 抵消**：补全复用路径生成的答案比直接 RAG 更长，
  output 单价是 input 的 4 倍，因此 token 省 40% 只换来成本省 5%。需要精简补全 prompt
  或提高直接复用比例才能把这部分收益拿回来
- **MiniMax-M2.5 等推理模型**：单次调用约 90+ reasoning tokens 计入 output 计费；
  `max_tokens` 不足时 `content` 会为空（代码已降级，但这部分 token 已经花掉了）
- **llama.cpp 线程安全**：默认占满 CPU 核，与 httpx/asyncio 频繁交替调用会随机段错误。
  已限制 `n_threads=1` 并加锁串行化 embed，若仍不稳可将后端切到 fastembed
- 未配置 `LLM_API_KEY`（LLM 未接入）时，Validator / 生成器走规则降级，指标不代表接入 LLM 后的真实收益
- 单元测试默认关闭真实 LLM（`use_real_llm=False`），否则每个用例都会打 API；
  需要验证 LLM 链路时用 `tests/test_pipeline.py` 末尾的「LLM 集成」分组
- L1 的编辑距离匹配仍是 O(n) 全表扫描（已加分词缓存与 O(1) 归一化索引）；缓存规模上万时应引入倒排索引
- L2 误复用无法仅靠相似度完全消除（应走 RAG 的问题与同义改写分布重叠），接入 LLM 后由 Validator 兜底
- 知识库为演示规模：8 条 FAQ / 8 篇文档 / 4 个商品

## 设计说明

Redis 后端通过 RedisVL 建立真正的 HNSW 向量索引执行 KNN 检索；若 Redis Stack 不可用则
回退为「Redis 存向量 + Python 端暴力扫描」，保证功能不中断。内存后端实现相同接口，
用于无外部服务时完整复现流程。
