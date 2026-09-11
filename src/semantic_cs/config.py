from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

# 在读取任何配置前先加载 .env（override=True 让 .env 覆盖运行环境可能残留的旧 LLM_* 变量）。
# 必须放在 config 导入的最早时机：get_settings() 带 lru_cache，若 .env 在它之后才被
# langchain_setup 加载，get_settings() 会缓存 Settings 默认值（MiniMax-M2.5），导致切模型不生效。
try:
    from dotenv import load_dotenv

    _ENV_FILE = ROOT_DIR / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE, override=True)
except Exception:
    pass


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    use_redis: bool = True
    # 真实维度由 embedding 后端决定，这里只作为哈希兜底的维度
    vector_dim: int = 384
    # auto | llama-cpp | fastembed | sentence-transformers | lsa | hashing
    embedding_backend: str = "auto"
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    # 本地 GGUF 向量模型（优先使用，避免联网下载）
    local_embedding_model: str = r"D:\vs_project\artical_knowledge\models\bge-m3-Q8_0.gguf"
    # 评测用：完整 RAG 链路单次提问的 LLM 调用数（无真实 LLM 时用于估算基线）
    rag_llm_calls_per_query: float = 2.0
    # 评测用：每次请求的缓存查询成本（USD），用于成本模型
    cache_lookup_cost: float = 0.00057
    l1_edit_threshold: float = 82.0
    l1_overlap_threshold: float = 0.65
    # 阈值由 scripts/calibrate_thresholds.py 在 bge-m3 上标定（含「业务内但 FAQ 无对应、
    # 应走 RAG」的负例）。正例 min=0.757、域外 max=0.529、应走RAG 均值=0.719，
    # 后两者与正例高度重叠，取精度优先：0.76 时召回 95.7%、误复用 3/11（F1 最优）。
    l2_similarity_threshold: float = 0.76
    retrieval_top_k: int = 4
    # Validator：>=0.80 直接复用（高精度区），0.76~0.80 先检索补全再复用，低于 0.76 回退完整 RAG
    validator_reuse_threshold: float = 0.80
    validator_complete_threshold: float = 0.76
    llm_call_cost: float = 0.002
    # 按 token 计费（USD / 每百万 token），DeepSeek 定价（deepseek-flash / v4-flash 同档）：
    # 输入约 $0.27~0.28/M、输出约 $1.10~1.14/M（此处取 0.28 / 1.14 估算）。
    # 接入 LLM 时成本按真实 token 计量，未接入时退化为「调用次数 × llm_call_cost」的估算。
    llm_input_cost_per_mtok: float = 0.28
    llm_output_cost_per_mtok: float = 1.14
    
    # LLM Configuration
    # provider: auto | deepseek | openai
    #   auto —— 按 base_url 判断（含 deepseek 用 ChatDeepSeek，否则 OpenAI 兼容接口）
    #   openai —— SiliconFlow / 其它 OpenAI 兼容网关，用 ChatOpenAI
    llm_provider: str = "auto"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "MiniMaxAI/MiniMax-M2.5"
    # 该系列多为推理模型，会额外消耗 reasoning tokens，max_tokens 需留足余量
    llm_max_tokens: int = 2048
    use_real_llm: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings(
        redis_url=os.getenv("SEMANTIC_CS_REDIS_URL", Settings.redis_url),
        use_redis=os.getenv("SEMANTIC_CS_USE_REDIS", "true").lower() != "false",
        embedding_backend=os.getenv("SEMANTIC_CS_EMBEDDING_BACKEND", Settings.embedding_backend),
        embedding_model_name=os.getenv(
            "SEMANTIC_CS_EMBEDDING_MODEL", Settings.embedding_model_name
        ),
        local_embedding_model=os.getenv(
            "SEMANTIC_CS_LOCAL_EMBEDDING_MODEL", Settings.local_embedding_model
        ),
        llm_provider=os.getenv("LLM_PROVIDER", Settings.llm_provider),
        # 通用 LLM_* 优先，其次兼容 DeepSeek 专用变量名
        llm_api_key=(
            os.getenv("LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or Settings.llm_api_key
        ),
        llm_base_url=(
            os.getenv("LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or Settings.llm_base_url
        ),
        llm_model=(
            os.getenv("LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or Settings.llm_model
        ),
        llm_max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", Settings.llm_max_tokens)),
        use_real_llm=os.getenv("USE_REAL_LLM", "true").lower() != "false",
    )
