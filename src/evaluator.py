from __future__ import annotations
from typing import Any, Dict, List, Tuple


def _avg_score(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(r["score"] for r in results) / len(results)


def _source_set(results: List[Dict[str, Any]]) -> set:
    return {r.get("source", "") for r in results}


def _modality_coverage(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    visual = sum(1 for r in results if r.get("has_image", False))
    return visual / len(results)


def compare_results(
    qwen_results: List[Dict[str, Any]],
    trad_results: List[Dict[str, Any]],
    qwen_time: float,
    trad_time: float,
) -> Dict[str, Any]:
    
    qwen_sources = _source_set(qwen_results)
    trad_sources = _source_set(trad_results)
    overlap = len(qwen_sources & trad_sources) / max(len(qwen_sources | trad_sources), 1)

    return {
        # Qwen
        "qwen_results":          qwen_results,
        "qwen_avg_score":        _avg_score(qwen_results),
        "qwen_time":             qwen_time,
        "qwen_modality_cov":     _modality_coverage(qwen_results),

        # Traditional
        "trad_results":          trad_results,
        "trad_avg_score":        _avg_score(trad_results),
        "trad_time":             trad_time,
        "trad_modality_cov":     _modality_coverage(trad_results),  # always 0

        # Cross-pipeline
        "result_overlap_ratio":  overlap,

        # Verdict
        "qwen_wins":  _avg_score(qwen_results) >= _avg_score(trad_results),
    }


def precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int,
) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / max(k, 1)


def mean_reciprocal_rank(
    retrieved: List[str],
    relevant: List[str],
) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def format_eval_report(
    comparison: Dict[str, Any],
    query: str,
) -> str:
    
    lines = [
        "## Retrieval Evaluation Report",
        f"**Query:** `{query}`",
        "",
        "| Metric | Qwen Multimodal | Traditional RAG |",
        "|---|---|---|",
        f"| Avg Similarity Score | {comparison['qwen_avg_score']:.4f} | {comparison['trad_avg_score']:.4f} |",
        f"| Latency (s) | {comparison['qwen_time']:.3f} | {comparison['trad_time']:.3f} |",
        f"| Modality Coverage | {comparison['qwen_modality_cov']*100:.1f}% | 0% |",
        f"| Result Overlap | {comparison['result_overlap_ratio']*100:.1f}% | — |",
        "",
        f"**Verdict:** {'Qwen outperforms' if comparison['qwen_wins'] else 'Traditional RAG scores higher on this query'}",
        "",
        "> *Note: Qwen's multimodal embeddings capture visual semantics that BM25/sentence-transformers miss.*",
    ]
    return "\n".join(lines)

