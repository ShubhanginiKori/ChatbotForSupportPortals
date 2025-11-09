from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping, Sequence

_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


def rerank_results(
    query: str,
    results: Sequence[Mapping[str, Any]],
    *,
    top_k: int | None = None,
    score_weight: float = 0.75,
) -> list[dict[str, Any]]:
    """
    Refine embedding matches by blending vector scores with quick lexical overlap.
    """

    if not results:
        return []

    if not 0.0 <= score_weight <= 1.0:
        raise ValueError("score_weight must be within the range [0.0, 1.0].")

    query_tokens = _tokenize(query)
    lexical_weight = 1.0 - score_weight

    unique_candidates: dict[str, tuple[Mapping[str, Any], float]] = {}
    for candidate in results:
        text = str(candidate.get("text", ""))
        base_score = float(candidate.get("score", 0.0))
        existing = unique_candidates.get(text)
        if existing is None or base_score > existing[1]:
            unique_candidates[text] = (candidate, base_score)

    reranked: list[dict[str, Any]] = []
    for text, (candidate, base_score) in unique_candidates.items():
        lexical_score = (
            _lexical_similarity(query_tokens, _tokenize(text))
            if query_tokens and lexical_weight > 0.0
            else 0.0
        )
        combined_score = (score_weight * base_score) + (lexical_weight * lexical_score)

        entry = dict(candidate)
        entry["text"] = text
        entry["score"] = base_score
        entry["lexical_score"] = lexical_score
        entry["rerank_score"] = combined_score
        reranked.append(entry)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    if top_k is not None and top_k > 0:
        reranked = reranked[:top_k]

    return reranked


def _tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in _WORD_PATTERN.findall(text)]
    filtered = [token for token in tokens if token not in _STOP_WORDS]
    if filtered:
        return filtered
    return tokens


def _lexical_similarity(
    query_tokens: Sequence[str], doc_tokens: Sequence[str]
) -> float:
    if not doc_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)
    shared = sum((query_counts & doc_counts).values())
    if shared == 0:
        return 0.0

    coverage = shared / len(query_tokens)
    unique_query = set(query_tokens)
    unique_doc = set(doc_tokens)
    intersection = len(unique_query & unique_doc)
    if intersection == 0:
        return coverage

    union = len(unique_query) + len(unique_doc) - intersection
    jaccard = intersection / union if union else 0.0

    return (0.7 * coverage) + (0.3 * jaccard)


if __name__ == "__main__":
    sample_results = [
        {"text": "Artificial intelligence transforms industries.", "score": 0.83},
        {"text": "Machine learning is a branch of AI.", "score": 0.78},
        {"text": "Cooking recipes require detailed instructions.", "score": 0.22},
    ]

    reranked_results = rerank_results("How is AI impacting industries?", sample_results)
    for item in reranked_results:
        print(item["text"], item["rerank_score"])
