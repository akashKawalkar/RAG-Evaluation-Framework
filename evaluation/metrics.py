
from typing import List, Set, Dict
import math

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / float(k)

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / float(len(relevant))

def hitatk(retrieved: List[str], relevant: Set[str], k: int) -> float:
    topk = retrieved[:k]
    return 1.0 if any(r in relevant for r in topk) else 0.0

def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for idx, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / float(idx)
    return 0.0

def dcgatk(relevances: List[float], k: int) -> float:
    if k <= 0:
        return 0.0
    s = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        denom = math.log2(i + 1.0)
        s += (rel / denom)
    return s

def ndcgatk(retrieved: List[str], relevant: Set[str], k: int) -> float:
    gains = [1.0 if r in relevant else 0.0 for r in retrieved]
    dcg = dcgatk(gains, k)
    ideal_gains = [1.0] * min(k, len(relevant))
    idcg = dcgatk(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def simple_mrr(retrieved: List[str], relevant: Set[str]) -> float:
    if not retrieved or not relevant:
        return 0.0
    for idx, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / float(idx)
    return 0.0