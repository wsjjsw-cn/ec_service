import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_cs.evaluation import OfflineEvaluator


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    # 不自建 pipeline：评测内部使用独立实例并在每轮前重置缓存，保证可复现
    report = OfflineEvaluator().run()
    print(f"total={report.total}")
    print(f"dynamic_intercepts={report.dynamic_intercepts}")
    print(f"cache_coverage_rate={pct(report.cache_coverage_rate)} (cold)")
    print(f"direct_cache_reuse_rate={pct(report.direct_cache_reuse_rate)} (cold)")
    print(f"  (warm) coverage={pct(report.warm_cache_coverage_rate)} "
          f"reuse={pct(report.warm_direct_cache_reuse_rate)}")
    print(f"avg_latency_ms={report.avg_latency_ms:.2f}")
    print(f"baseline_latency_ms={report.baseline_latency_ms:.2f}")
    print(f"latency_reduction_rate={pct(report.latency_reduction_rate)}")
    print(f"throughput_multiplier={report.throughput_multiplier:.2f}x")
    print(f"avg_llm_saved={report.avg_llm_saved:.2f}")
    print(f"total_llm_saved={report.total_llm_saved}")
    print(f"cost_saving_rate={pct(report.cost_saving_rate)}")
    print()
    print(f"embedding_backend={report.embedding_backend}")
    print(f"llm_enabled={report.llm_enabled}")
    print(f"llm_calls baseline={report.llm_calls_baseline} actual={report.llm_calls_actual}")
    print(f"tokens baseline={report.baseline_input_tokens}+{report.baseline_output_tokens} "
          f"actual={report.actual_input_tokens}+{report.actual_output_tokens}")
    print(f"cost baseline=${report.baseline_cost_usd:.6f} actual=${report.actual_cost_usd:.6f}")
    for note in report.notes:
        print(f"note: {note}")


if __name__ == "__main__":
    main()
