"""阈值校准：用标注的同义改写/无关问句对，为 bge-m3 重新标定 L2 与 Validator 阈值。

旧阈值（L2=0.78, reuse=0.84, complete=0.70）是在 md5 哈希词向量下标定的，
换成真实语义模型后分布整体变化，必须重新标定，否则要么全漏要么全中。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_cs.data_loader import load_faqs
from semantic_cs.text import EmbeddingModel

OUT = "OUT"  # 域外问题：任何 FAQ 都不应命中
RAG = "RAG"  # 业务域内但 FAQ 无对应条目：应走完整 RAG，被缓存复用即误报

# (改写问句, 期望)。期望为 FAQ id 表示应复用该条目；OUT 表示域外；RAG 表示应走 RAG。
LABELS: list[tuple[str, str]] = [
    # --- 退货 ---
    ("退货需要什么条件", "faq_return_7d"),
    ("不想要了能退吗", "faq_return_7d"),
    ("七天无理由退货的要求是什么", "faq_return_7d"),
    ("拆封了还能退不", "faq_return_7d"),
    # --- 退款 ---
    ("退款多久能到", "faq_refund_time"),
    ("退的钱什么时候到账", "faq_refund_time"),
    ("钱什么时候退回来", "faq_refund_time"),
    ("退款要等几天", "faq_refund_time"),
    # --- 发票 ---
    ("发票怎么开", "faq_invoice"),
    ("怎么开发票", "faq_invoice"),
    ("要发票的话怎么弄", "faq_invoice"),
    # --- 物流 ---
    ("下单后几天能到", "faq_shipping"),
    ("多久能发货", "faq_shipping"),
    ("物流要几天", "faq_shipping"),
    # --- 价保 ---
    ("降价了能补差价吗", "faq_price_protection"),
    ("买贵了能退钱吗", "faq_price_protection"),
    ("价保怎么申请", "faq_price_protection"),
    # --- 换货 ---
    ("可以换个尺码吗", "faq_exchange"),
    ("换货怎么操作", "faq_exchange"),
    ("能换一个吗", "faq_exchange"),
    # --- 积分 ---
    ("积分怎么用", "faq_member_points"),
    ("积分能干嘛", "faq_member_points"),
    ("积分可以抵钱吗", "faq_member_points"),
    # --- 取消订单 ---
    ("怎么取消订单", "faq_cancel"),
    ("能取消订单吗", "faq_cancel"),
    ("订单不要了怎么取消", "faq_cancel"),
    # --- 域外：不应命中任何 FAQ，命中即为误报 ---
    ("今天天气怎么样", OUT),
    ("讲个笑话", OUT),
    ("推荐一部电影", OUT),
    ("你是谁", OUT),
    ("帮我写一首诗", OUT),
    ("怎么学习英语", OUT),
    # --- 业务域内但 FAQ 无对应条目：应该走完整 RAG，被缓存复用会给出偏题答案 ---
    ("优惠券过期了退款会退回来吗？", RAG),
    ("增值税专票需要填写什么？", RAG),
    ("积分可以提现吗？", RAG),
    ("优惠券使用有什么限制？", RAG),
    ("保修需要提交什么材料？", RAG),
    ("人为损坏可以免费维修吗？", RAG),
    ("组合支付退款怎么退？", RAG),
    ("优惠券取消订单后会返还吗？", RAG),
    ("电子普通发票支持个人抬头吗？", RAG),
    ("大促期间物流会延迟吗？", RAG),
    ("不喜欢可以退货吗，包装拆了", RAG),
]


def main() -> None:
    faqs = load_faqs()
    model = EmbeddingModel(384)
    print(f"backend={model.backend_name} dim={model.dim}\n")

    faq_vecs = {faq.id: model.encode(faq.question) for faq in faqs}

    def best_match(question: str) -> tuple[str, float]:
        vec = model.encode(question)
        scored = [
            (faq.id, sum(a * b for a, b in zip(vec, faq_vecs[faq.id])))
            for faq in faqs
        ]
        return max(scored, key=lambda item: item[1])

    positive_correct: list[float] = []
    positive_wrong: list[tuple[str, str, float]] = []
    negative_scores: list[float] = []  # 域外
    rag_scores: list[float] = []  # 业务内但应走 RAG

    print(f"{'改写问句':<24}{'期望':<22}{'实际命中':<22}{'分数':>7}  判定")
    print("-" * 90)
    for question, expected in LABELS:
        matched_id, score = best_match(question)
        if expected == OUT:
            negative_scores.append(score)
            verdict = "OK(域外被拒)" if score < 0.78 else "需拒"
            print(f"{question:<24}{'(域外)':<22}{matched_id:<22}{score:>7.3f}  {verdict}")
        elif expected == RAG:
            rag_scores.append(score)
            verdict = "OK(走RAG)" if score < 0.65 else "误复用"
            print(f"{question:<24}{'(应走RAG)':<22}{matched_id:<22}{score:>7.3f}  {verdict}")
        elif matched_id == expected:
            positive_correct.append(score)
            print(f"{question:<24}{expected:<22}{matched_id:<22}{score:>7.3f}  命中正确")
        else:
            positive_wrong.append((question, matched_id, score))
            print(f"{question:<24}{expected:<22}{matched_id:<22}{score:>7.3f}  命中错误")

    print("\n=== 分布 ===")
    print(f"正例命中正确: n={len(positive_correct)} min={min(positive_correct):.3f} "
          f"max={max(positive_correct):.3f} mean={sum(positive_correct)/len(positive_correct):.3f}")
    print(f"域外问句最高分: n={len(negative_scores)} min={min(negative_scores):.3f} "
          f"max={max(negative_scores):.3f}")
    print(f"应走RAG问句最高分: n={len(rag_scores)} min={min(rag_scores):.3f} "
          f"max={max(rag_scores):.3f} "
          f"mean={sum(rag_scores)/len(rag_scores):.3f}")
    if positive_wrong:
        print("命中错误的正例：")
        for question, matched, score in positive_wrong:
            print(f"  {question} -> {matched} ({score:.3f})")

    print("\n=== 阈值扫描（L2 候选阈值 -> 召回/误报）===")
    print(f"{'阈值':>6}{'正确召回':>10}{'域外误报':>10}{'RAG误复用':>12}{'F1':>8}")
    best = (0.0, -1.0)
    for threshold in [x / 100 for x in range(50, 91, 2)]:
        tp = sum(1 for s in positive_correct if s >= threshold)
        fp_out = sum(1 for s in negative_scores if s >= threshold)
        fp_rag = sum(1 for s in rag_scores if s >= threshold)
        fp = fp_out + fp_rag
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / len(positive_correct)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{threshold:>6.2f}{recall:>10.2%}{fp_out:>10d}{fp_rag:>12d}{f1:>8.3f}")
        if f1 > best[1]:
            best = (threshold, f1)
    print(f"\n推荐 L2 阈值 = {best[0]:.2f} (F1={best[1]:.3f})")


if __name__ == "__main__":
    main()
