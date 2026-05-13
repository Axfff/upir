import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            out[i] += value
    inv = 1.0 / len(vectors)
    return [x * inv for x in out]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_norm(a: Sequence[float]) -> float:
    return math.sqrt(max(1e-12, dot(a, a)))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return dot(a, b) / (l2_norm(a) * l2_norm(b))


def softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    t = max(1e-8, temperature)
    scaled = [x / t for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [x / s for x in exps]


def logsumexp(values: Sequence[float], temperature: float = 1.0) -> float:
    t = max(1e-8, temperature)
    scaled = [x / t for x in values]
    m = max(scaled)
    total = sum(math.exp(x - m) for x in scaled)
    return t * (m + math.log(max(total, 1e-12)))


def vector_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def vector_scale(a: Sequence[float], s: float) -> List[float]:
    return [x * s for x in a]


def weighted_sum(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
    dim = len(vectors[0]) if vectors else 0
    out = [0.0] * dim
    for w, vec in zip(weights, vectors):
        for i, value in enumerate(vec):
            out[i] += w * value
    return out


def chunked(iterable: Sequence, chunk_size: int) -> Iterable[Sequence]:
    for start in range(0, len(iterable), chunk_size):
        yield iterable[start : start + chunk_size]


def random_unit_vector(dim: int, rng: random.Random) -> List[float]:
    vals = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = l2_norm(vals)
    return [x / norm for x in vals]


def memory_bytes_for_vectors(vectors: Sequence[Sequence[float]]) -> int:
    # Conservative estimate: float64 in Python-side metrics output.
    if not vectors:
        return 0
    return len(vectors) * len(vectors[0]) * 8


def entropy(probabilities: Sequence[float]) -> float:
    out = 0.0
    for p in probabilities:
        if p > 1e-12:
            out -= p * math.log(p)
    return out


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    items = sorted(values)
    n = len(items)
    mid = n // 2
    if n % 2 == 1:
        return items[mid]
    return 0.5 * (items[mid - 1] + items[mid])
