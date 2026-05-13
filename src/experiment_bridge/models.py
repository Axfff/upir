import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Ligand, Pocket
from .utils import (
    cosine,
    dot,
    entropy,
    l2_norm,
    logsumexp,
    mean_vector,
    random_unit_vector,
    softmax,
    weighted_sum,
)


@dataclass
class ModelConfig:
    seed: int
    tau_match: float = 0.2
    tau_assign: float = 0.7
    attn_scale: float = 1.5
    k_slots: int = 4
    t_slots: int = 4
    epochs: int = 25
    lr: float = 0.3
    weight_decay: float = 1e-4
    batch_size: int = 1024


@dataclass
class Calibrator:
    alpha: float
    beta: float
    train_loss: List[float]


def _stable_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _late_interaction(vectors_a: Sequence[Sequence[float]], vectors_b: Sequence[Sequence[float]], tau_match: float) -> float:
    if not vectors_a or not vectors_b:
        return -1e6
    per_a: List[float] = []
    for va in vectors_a:
        per_a.append(logsumexp([dot(va, vb) for vb in vectors_b], temperature=tau_match))
    return sum(per_a) / len(per_a)


def _colbert_maxsim(vectors_a: Sequence[Sequence[float]], vectors_b: Sequence[Sequence[float]]) -> float:
    if not vectors_a or not vectors_b:
        return -1e6
    per_a: List[float] = []
    for va in vectors_a:
        per_a.append(max(dot(va, vb) for vb in vectors_b))
    return sum(per_a) / len(per_a)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _adaptive_collapse_signal(diag: Optional[Dict[str, float]]) -> float:
    if not diag:
        return 0.5
    balance = _clamp(float(diag.get("slot_balance", 0.5)), 0.0, 1.0)
    inter = _clamp((float(diag.get("inter_slot_cosine", 0.0)) + 1.0) / 2.0, 0.0, 1.0)
    # Empirically robust proxy for broad-pocket EF5 regressions.
    inter_std = _clamp(float(diag.get("inter_slot_cosine_std", 0.01)) / 0.02, 0.0, 1.0)
    assign = _clamp(float(diag.get("assignment_entropy", 0.0)) / max(1e-8, math.log(4.0)), 0.0, 1.0)
    # Higher means more collapsed/ambiguous slot assignment.
    return _clamp(0.45 * inter_std + 0.25 * (1.0 - balance) + 0.20 * inter + 0.10 * assign, 0.0, 1.0)


def _adaptive_late_weight(pocket_diag: Optional[Dict[str, float]], ligand_diag: Optional[Dict[str, float]]) -> float:
    p = _adaptive_collapse_signal(pocket_diag)
    l = _adaptive_collapse_signal(ligand_diag)
    collapse = _clamp(0.7 * p + 0.3 * l, 0.0, 1.0)
    # Collapse -> rely more on robust max-sim fallback.
    return _clamp(0.90 - 0.55 * collapse, 0.30, 0.90)


def _attention_pool(tokens: Sequence[Sequence[float]], scale: float) -> List[float]:
    if not tokens:
        return []
    logits = [scale * l2_norm(tok) for tok in tokens]
    weights = softmax(logits)
    return weighted_sum(tokens, weights)


def _build_queries(dim: int, slots: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    return [random_unit_vector(dim, rng) for _ in range(slots)]


def _compress_tokens(
    entity_id: str,
    tokens: Sequence[Sequence[float]],
    slots: int,
    tau_assign: float,
    seed: int,
) -> Tuple[List[List[float]], Dict[str, float]]:
    if not tokens:
        return [], {
            "assignment_entropy": 0.0,
            "slot_balance": 0.0,
            "inter_slot_cosine": 0.0,
            "inter_slot_cosine_std": 0.0,
        }

    dim = len(tokens[0])
    qseed = _stable_seed(f"{entity_id}:{seed}:{slots}")
    queries = _build_queries(dim=dim, slots=slots, seed=qseed)

    slot_sums = [[0.0] * dim for _ in range(slots)]
    slot_weight_totals = [0.0] * slots
    token_entropies: List[float] = []

    for tok in tokens:
        logits = [dot(tok, q) for q in queries]
        weights = softmax(logits, temperature=tau_assign)
        token_entropies.append(entropy(weights))
        for k, wk in enumerate(weights):
            slot_weight_totals[k] += wk
            for d in range(dim):
                slot_sums[k][d] += wk * tok[d]

    summaries: List[List[float]] = []
    for k in range(slots):
        denom = max(1e-8, slot_weight_totals[k])
        summaries.append([v / denom for v in slot_sums[k]])

    norm_total = sum(slot_weight_totals)
    probs = [x / max(1e-12, norm_total) for x in slot_weight_totals]
    slot_balance = entropy(probs) / max(1e-12, math.log(max(2, slots)))

    cos_vals: List[float] = []
    for i in range(slots):
        for j in range(i + 1, slots):
            cos_vals.append(cosine(summaries[i], summaries[j]))
    inter_slot_cosine = sum(cos_vals) / len(cos_vals) if cos_vals else 1.0
    if cos_vals:
        var = sum((x - inter_slot_cosine) ** 2 for x in cos_vals) / len(cos_vals)
        inter_slot_cosine_std = math.sqrt(var)
    else:
        inter_slot_cosine_std = 0.0

    diagnostics = {
        "assignment_entropy": sum(token_entropies) / len(token_entropies),
        "slot_balance": slot_balance,
        "inter_slot_cosine": inter_slot_cosine,
        "inter_slot_cosine_std": inter_slot_cosine_std,
    }
    return summaries, diagnostics


def encode_pocket(model_name: str, pocket: Pocket, cfg: ModelConfig) -> Tuple[List[List[float]], Dict[str, float]]:
    pocket_tokens = pocket.materialize_tokens()
    if model_name == "B0":
        return [mean_vector(pocket_tokens)], {}
    if model_name == "B1":
        return [_attention_pool(pocket_tokens, scale=cfg.attn_scale)], {}
    if model_name == "U1":
        return pocket_tokens, {}
    if model_name == "B2_COLBERT":
        return pocket_tokens, {}
    if model_name in {"M1", "M2_ADAPTIVE"}:
        compressed, diag = _compress_tokens(
            entity_id=pocket.pid,
            tokens=pocket_tokens,
            slots=cfg.k_slots,
            tau_assign=cfg.tau_assign,
            seed=cfg.seed,
        )
        return compressed, diag
    if model_name == "B3_SLOT_COS":
        compressed, diag = _compress_tokens(
            entity_id=pocket.pid,
            tokens=pocket_tokens,
            slots=cfg.k_slots,
            tau_assign=cfg.tau_assign,
            seed=cfg.seed,
        )
        return compressed, diag
    if model_name == "B4_SLOT_MAXSIM":
        compressed, diag = _compress_tokens(
            entity_id=pocket.pid,
            tokens=pocket_tokens,
            slots=cfg.k_slots,
            tau_assign=cfg.tau_assign,
            seed=cfg.seed,
        )
        return compressed, diag
    raise ValueError(f"Unsupported model: {model_name}")


def encode_ligand(model_name: str, ligand: Ligand, cfg: ModelConfig) -> Tuple[List[List[float]], Dict[str, float]]:
    ligand_tokens = ligand.materialize_tokens()
    if model_name == "B0":
        return [mean_vector(ligand_tokens)], {}
    if model_name == "B1":
        return [_attention_pool(ligand_tokens, scale=cfg.attn_scale)], {}
    if model_name == "U1":
        return ligand_tokens, {}
    if model_name == "B2_COLBERT":
        return ligand_tokens, {}
    if model_name in {"M1", "M2_ADAPTIVE"}:
        compressed, diag = _compress_tokens(
            entity_id=ligand.lid,
            tokens=ligand_tokens,
            slots=cfg.t_slots,
            tau_assign=cfg.tau_assign,
            seed=cfg.seed + 701,
        )
        return compressed, diag
    if model_name == "B3_SLOT_COS":
        compressed, diag = _compress_tokens(
            entity_id=ligand.lid,
            tokens=ligand_tokens,
            slots=cfg.t_slots,
            tau_assign=cfg.tau_assign,
            seed=cfg.seed + 701,
        )
        return compressed, diag
    if model_name == "B4_SLOT_MAXSIM":
        compressed, diag = _compress_tokens(
            entity_id=ligand.lid,
            tokens=ligand_tokens,
            slots=cfg.t_slots,
            tau_assign=cfg.tau_assign,
            seed=cfg.seed + 701,
        )
        return compressed, diag
    raise ValueError(f"Unsupported model: {model_name}")


def pair_raw_score(
    model_name: str,
    pocket_repr: Sequence[Sequence[float]],
    ligand_repr: Sequence[Sequence[float]],
    cfg: ModelConfig,
    pocket_diag: Optional[Dict[str, float]] = None,
    ligand_diag: Optional[Dict[str, float]] = None,
) -> float:
    if model_name in {"B0", "B1"}:
        return cosine(pocket_repr[0], ligand_repr[0])
    if model_name in {"U1", "M1"}:
        return _late_interaction(pocket_repr, ligand_repr, tau_match=cfg.tau_match)
    if model_name == "M2_ADAPTIVE":
        late_w = _adaptive_late_weight(pocket_diag, ligand_diag)
        collapse = _clamp(1.0 - late_w, 0.0, 1.0)
        # Softer matching for collapsed-slot pockets to reduce EF@5 regressions.
        tau_eff = cfg.tau_match * (1.0 + 1.25 * collapse)
        late = _late_interaction(pocket_repr, ligand_repr, tau_match=tau_eff)
        robust = _colbert_maxsim(pocket_repr, ligand_repr)
        return late_w * late + (1.0 - late_w) * robust
    if model_name == "B3_SLOT_COS":
        return cosine(mean_vector(pocket_repr), mean_vector(ligand_repr))
    if model_name == "B4_SLOT_MAXSIM":
        return _colbert_maxsim(pocket_repr, ligand_repr)
    if model_name == "B2_COLBERT":
        return _colbert_maxsim(pocket_repr, ligand_repr)
    raise ValueError(f"Unsupported model: {model_name}")


def fit_calibrator(raw_scores: List[float], labels: List[int], cfg: ModelConfig) -> Calibrator:
    alpha = 1.0
    beta = 0.0
    train_loss: List[float] = []

    pairs = list(zip(raw_scores, labels))
    rng = random.Random(cfg.seed + 999)

    for _ in range(cfg.epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        n = 0
        for start in range(0, len(pairs), cfg.batch_size):
            batch = pairs[start : start + cfg.batch_size]
            grad_a = 0.0
            grad_b = 0.0
            for raw, y in batch:
                s = alpha * raw + beta
                s = max(min(s, 50.0), -50.0)
                p = 1.0 / (1.0 + math.exp(-s))
                p = min(max(p, 1e-8), 1.0 - 1e-8)
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
                grad = p - y
                grad_a += grad * raw
                grad_b += grad
                n += 1
            bs = max(1, len(batch))
            alpha -= cfg.lr * (grad_a / bs + cfg.weight_decay * alpha)
            beta -= cfg.lr * (grad_b / bs)
            # Keep ranking monotonic with the raw model score.
            if alpha < 1e-4:
                alpha = 1e-4

        train_loss.append(total_loss / max(1, n))

    return Calibrator(alpha=alpha, beta=beta, train_loss=train_loss)


def apply_calibrator(raw_score: float, calibrator: Calibrator) -> float:
    return calibrator.alpha * raw_score + calibrator.beta
