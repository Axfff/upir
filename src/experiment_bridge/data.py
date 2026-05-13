import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .text_tokens import stable_seed, text_to_tokens
from .utils import dot, ensure_dir, load_json, logsumexp, random_unit_vector, save_json


@dataclass
class Pocket:
    pid: str
    tokens: Optional[List[List[float]]] = None
    text: Optional[str] = None
    token_dim: int = 0
    token_kmer: int = 0
    token_max_tokens: int = 0
    token_base_seed: int = 0

    def materialize_tokens(self) -> List[List[float]]:
        if self.tokens is not None:
            return self.tokens
        if not self.text:
            return []
        return text_to_tokens(
            text=self.text,
            dim=self.token_dim,
            k=self.token_kmer,
            max_tokens=self.token_max_tokens,
            base_seed=self.token_base_seed,
        )


@dataclass
class Ligand:
    lid: str
    scaffold: str
    tokens: Optional[List[List[float]]] = None
    text: Optional[str] = None
    token_dim: int = 0
    token_kmer: int = 0
    token_max_tokens: int = 0
    token_base_seed: int = 0

    def materialize_tokens(self) -> List[List[float]]:
        if self.tokens is not None:
            return self.tokens
        if not self.text:
            return []
        return text_to_tokens(
            text=self.text,
            dim=self.token_dim,
            k=self.token_kmer,
            max_tokens=self.token_max_tokens,
            base_seed=self.token_base_seed,
        )


@dataclass
class DatasetBundle:
    name: str
    dim: int
    pockets: List[Pocket]
    ligands: List[Ligand]
    labels: Dict[str, Dict[str, int]]
    meta: Optional[Dict] = None
    source_path: Optional[str] = None


@dataclass
class SplitView:
    split_name: str
    train_pairs: List[Tuple[str, str, int]]
    eval_pocket_ids: List[str]
    candidate_ligand_ids: List[str]
    notes: str


def get_primary_query_rule(bundle: DatasetBundle) -> Dict[str, int]:
    if not bundle.meta:
        return {"min_known_pos": 1, "min_known_neg": 0}
    rule = bundle.meta.get("primary_evaluable_query_rule", {}) or {}
    return {
        "min_known_pos": int(rule.get("min_known_pos", 1)),
        "min_known_neg": int(rule.get("min_known_neg", 0)),
    }


def _make_tokens(base: List[float], count: int, rng: random.Random, noise: float) -> List[List[float]]:
    tokens: List[List[float]] = []
    for _ in range(count):
        tokens.append([v + rng.gauss(0.0, noise) for v in base])
    return tokens


def _true_score(pocket_tokens: List[List[float]], ligand_tokens: List[List[float]], tau: float) -> float:
    per_residue = []
    for p in pocket_tokens:
        per_residue.append(logsumexp([dot(p, l) for l in ligand_tokens], temperature=tau))
    return sum(per_residue) / max(1, len(per_residue))


def generate_synthetic_dataset(
    dataset_name: str,
    seed: int,
    dim: int,
    num_pockets: int,
    num_ligands: int,
    pocket_tokens: int,
    ligand_tokens: int,
    positive_rate: float,
    true_tau: float,
) -> DatasetBundle:
    rng = random.Random(seed)

    chemotypes = [random_unit_vector(dim, rng) for _ in range(12)]

    pockets: List[Pocket] = []
    ligands: List[Ligand] = []

    # Pockets prefer 2-3 chemotypes each; tokens include sub-pocket heterogeneity.
    pocket_pref: Dict[str, List[int]] = {}
    for i in range(num_pockets):
        pid = f"P{i:04d}"
        pref = rng.sample(range(len(chemotypes)), k=3)
        pocket_pref[pid] = pref
        base = [0.0] * dim
        for idx in pref:
            for d in range(dim):
                base[d] += chemotypes[idx][d]
        base = [x / len(pref) for x in base]
        pockets.append(Pocket(pid=pid, tokens=_make_tokens(base, pocket_tokens, rng, noise=0.35)))

    for j in range(num_ligands):
        lid = f"L{j:05d}"
        scaffold_id = f"S{j % 10}"
        cidx = rng.randrange(len(chemotypes))
        base = chemotypes[cidx]
        ligands.append(Ligand(lid=lid, scaffold=scaffold_id, tokens=_make_tokens(base, ligand_tokens, rng, noise=0.42)))

    labels: Dict[str, Dict[str, int]] = {}
    for p in pockets:
        raw_scores: List[Tuple[str, float]] = []
        for l in ligands:
            s = _true_score(p.tokens, l.tokens, tau=true_tau)
            # pocket-preference bonus to create one-to-many compatibility.
            cidx = int(l.lid[1:]) % len(chemotypes)
            if cidx in pocket_pref[p.pid]:
                s += 0.8
            raw_scores.append((l.lid, s))
        raw_scores.sort(key=lambda x: x[1], reverse=True)
        positives = max(2, int(len(raw_scores) * positive_rate))
        pos_set = {lid for lid, _ in raw_scores[:positives]}
        labels[p.pid] = {lid: (1 if lid in pos_set else 0) for lid, _ in raw_scores}

    return DatasetBundle(name=dataset_name, dim=dim, pockets=pockets, ligands=ligands, labels=labels)


def save_dataset(path: Path, bundle: DatasetBundle) -> None:
    ensure_dir(path.parent)
    payload = {
        "name": bundle.name,
        "dim": bundle.dim,
        "pockets": [{"pid": p.pid, "tokens": p.tokens} for p in bundle.pockets],
        "ligands": [{"lid": l.lid, "tokens": l.tokens, "scaffold": l.scaffold} for l in bundle.ligands],
        "labels": bundle.labels,
    }
    if bundle.meta is not None:
        payload["meta"] = bundle.meta
    save_json(path, payload)


def _token_cfg(item: Dict, payload: Dict, default_kind: str) -> Dict:
    cfg = dict(item.get("tokenization", {}))
    if cfg:
        return cfg
    meta_cfg = payload.get("meta", {}).get("tokenization", {})
    kinds = meta_cfg.get("kinds", {})
    text_kind = item.get("text_kind", default_kind)
    kind_cfg = dict(kinds.get(text_kind, {}))
    if not kind_cfg:
        return {}
    kind_cfg["dim"] = meta_cfg.get("dim", payload.get("dim", 0))
    return kind_cfg


def load_dataset(path: Path) -> DatasetBundle:
    payload = load_json(path)
    pockets: List[Pocket] = []
    for item in payload["pockets"]:
        cfg = _token_cfg(item, payload, default_kind="pocket")
        pockets.append(
            Pocket(
                pid=item["pid"],
                tokens=item.get("tokens"),
                text=item.get("text"),
                token_dim=int(cfg.get("dim", payload["dim"])),
                token_kmer=int(cfg.get("k", 0)),
                token_max_tokens=int(cfg.get("max_tokens", 0)),
                token_base_seed=int(cfg.get("base_seed", 0)),
            )
        )

    ligands: List[Ligand] = []
    for item in payload["ligands"]:
        cfg = _token_cfg(item, payload, default_kind="ligand")
        ligands.append(
            Ligand(
                lid=item["lid"],
                tokens=item.get("tokens"),
                scaffold=item.get("scaffold", "S0"),
                text=item.get("text"),
                token_dim=int(cfg.get("dim", payload["dim"])),
                token_kmer=int(cfg.get("k", 0)),
                token_max_tokens=int(cfg.get("max_tokens", 0)),
                token_base_seed=int(cfg.get("base_seed", 0)),
            )
        )
    return DatasetBundle(
        name=payload.get("name", path.stem),
        dim=payload["dim"],
        pockets=pockets,
        ligands=ligands,
        labels=payload["labels"],
        meta=payload.get("meta"),
        source_path=str(path),
    )


def load_or_create_dataset(
    path: Path,
    dataset_name: str,
    seed: int,
    dim: int,
    num_pockets: int,
    num_ligands: int,
    pocket_tokens: int,
    ligand_tokens: int,
    positive_rate: float,
    true_tau: float,
) -> DatasetBundle:
    if path.exists():
        return load_dataset(path)
    bundle = generate_synthetic_dataset(
        dataset_name=dataset_name,
        seed=seed,
        dim=dim,
        num_pockets=num_pockets,
        num_ligands=num_ligands,
        pocket_tokens=pocket_tokens,
        ligand_tokens=ligand_tokens,
        positive_rate=positive_rate,
        true_tau=true_tau,
    )
    save_dataset(path, bundle)
    return bundle


def _load_protocol(bundle: DatasetBundle) -> Optional[Dict]:
    if not bundle.meta:
        return None
    path_hint = bundle.meta.get("protocol_path")
    if not path_hint:
        return None
    base = Path(bundle.source_path).parent if bundle.source_path else Path(".")
    protocol_path = Path(path_hint)
    if not protocol_path.is_absolute():
        protocol_path = base / protocol_path
    if not protocol_path.exists():
        return None
    return load_json(protocol_path)


def _build_protocol_standard_split(bundle: DatasetBundle, protocol: Dict, seed: int) -> SplitView:
    cfg = protocol["splits"]["standard"]
    num_folds = int(cfg.get("num_folds", 5))
    fold = seed % max(1, num_folds)
    eval_pocket_ids = [p.pid for p in bundle.pockets]
    candidate_ligand_ids = [l.lid for l in bundle.ligands]
    train_pairs: List[Tuple[str, str, int]] = []
    for pid, row in bundle.labels.items():
        for lid, label in row.items():
            pair_fold = stable_seed(f"{pid}|{lid}") % num_folds
            if pair_fold != fold:
                train_pairs.append((pid, lid, label))
    notes = f"predefined standard split fold {fold}/{num_folds} from benchmark protocol"
    return SplitView(
        split_name="standard",
        train_pairs=train_pairs,
        eval_pocket_ids=eval_pocket_ids,
        candidate_ligand_ids=candidate_ligand_ids,
        notes=notes,
    )


def _build_protocol_target_split(bundle: DatasetBundle, protocol: Dict, seed: int) -> SplitView:
    folds = protocol["splits"]["target"]["folds"]
    fold = folds[seed % len(folds)]
    eval_pocket_ids = list(fold["eval_query_ids"])
    train_pockets = {p.pid for p in bundle.pockets} - set(eval_pocket_ids)
    candidate_ligand_ids = [l.lid for l in bundle.ligands]
    train_pairs: List[Tuple[str, str, int]] = []
    for pid in train_pockets:
        for lid, label in bundle.labels.get(pid, {}).items():
            train_pairs.append((pid, lid, label))
    notes = f"predefined target split fold {fold['fold_id']} from benchmark protocol"
    return SplitView(
        split_name="target",
        train_pairs=train_pairs,
        eval_pocket_ids=eval_pocket_ids,
        candidate_ligand_ids=candidate_ligand_ids,
        notes=notes,
    )


def _build_protocol_target_rec_cluster_split(bundle: DatasetBundle, protocol: Dict, seed: int) -> SplitView:
    folds = protocol["splits"]["target_rec_cluster"]["folds"]
    fold = folds[seed % len(folds)]
    eval_pocket_ids = list(fold["eval_query_ids"])
    train_pockets = {p.pid for p in bundle.pockets} - set(eval_pocket_ids)
    candidate_ligand_ids = [l.lid for l in bundle.ligands]
    train_pairs: List[Tuple[str, str, int]] = []
    for pid in train_pockets:
        for lid, label in bundle.labels.get(pid, {}).items():
            train_pairs.append((pid, lid, label))
    notes = f"predefined target_rec_cluster split fold {fold['fold_id']} from benchmark protocol"
    return SplitView(
        split_name="target_rec_cluster",
        train_pairs=train_pairs,
        eval_pocket_ids=eval_pocket_ids,
        candidate_ligand_ids=candidate_ligand_ids,
        notes=notes,
    )


def _build_protocol_scaffold_split(bundle: DatasetBundle, protocol: Dict, seed: int) -> SplitView:
    folds = protocol["splits"]["scaffold"]["folds"]
    fold = folds[seed % len(folds)]
    eval_scaffolds = set(fold["eval_scaffolds"])
    train_pairs: List[Tuple[str, str, int]] = []
    eval_pocket_ids = [p.pid for p in bundle.pockets]
    candidate_ligand_ids = [l.lid for l in bundle.ligands if l.scaffold in eval_scaffolds]
    train_lids = {l.lid for l in bundle.ligands if l.scaffold not in eval_scaffolds}
    for pid in eval_pocket_ids:
        for lid in train_lids:
            if lid in bundle.labels.get(pid, {}):
                train_pairs.append((pid, lid, bundle.labels[pid][lid]))
    notes = f"predefined scaffold split fold {fold['fold_id']} from benchmark protocol"
    return SplitView(
        split_name="scaffold",
        train_pairs=train_pairs,
        eval_pocket_ids=eval_pocket_ids,
        candidate_ligand_ids=candidate_ligand_ids,
        notes=notes,
    )


def _build_protocol_joint_split(bundle: DatasetBundle, protocol: Dict, seed: int) -> SplitView:
    folds = protocol["splits"]["joint_ood"]["folds"]
    fold = folds[seed % len(folds)]
    eval_pocket_ids = list(fold["eval_query_ids"])
    eval_scaffolds = set(fold["eval_scaffolds"])
    train_pockets = {p.pid for p in bundle.pockets} - set(eval_pocket_ids)
    train_lids = {l.lid for l in bundle.ligands if l.scaffold not in eval_scaffolds}
    candidate_ligand_ids = [l.lid for l in bundle.ligands if l.scaffold in eval_scaffolds]
    train_pairs: List[Tuple[str, str, int]] = []
    for pid in train_pockets:
        for lid in train_lids:
            if lid in bundle.labels.get(pid, {}):
                train_pairs.append((pid, lid, bundle.labels[pid][lid]))
    notes = f"predefined joint_ood split fold {fold['fold_id']} from benchmark protocol"
    return SplitView(
        split_name="joint_ood",
        train_pairs=train_pairs,
        eval_pocket_ids=eval_pocket_ids,
        candidate_ligand_ids=candidate_ligand_ids,
        notes=notes,
    )


def build_split_view(bundle: DatasetBundle, split_name: str, seed: int) -> SplitView:
    protocol = _load_protocol(bundle)
    if protocol and split_name in protocol.get("splits", {}):
        if split_name == "standard":
            return _build_protocol_standard_split(bundle, protocol, seed)
        if split_name == "target":
            return _build_protocol_target_split(bundle, protocol, seed)
        if split_name == "target_rec_cluster":
            return _build_protocol_target_rec_cluster_split(bundle, protocol, seed)
        if split_name == "scaffold":
            return _build_protocol_scaffold_split(bundle, protocol, seed)
        if split_name == "joint_ood":
            return _build_protocol_joint_split(bundle, protocol, seed)

    rng = random.Random(seed + 33)
    pocket_ids = [p.pid for p in bundle.pockets]
    ligand_ids = [l.lid for l in bundle.ligands]

    train_pairs: List[Tuple[str, str, int]] = []
    eval_pocket_ids: List[str] = []
    candidate_ligand_ids: List[str] = []

    if split_name not in {"standard", "scaffold", "target", "target_rec_cluster", "joint_ood"}:
        raise ValueError(f"Unsupported split: {split_name}")

    if split_name == "target":
        shuffled = pocket_ids[:]
        rng.shuffle(shuffled)
        cut = max(1, int(0.8 * len(shuffled)))
        train_pockets = set(shuffled[:cut])
        eval_pocket_ids = shuffled[cut:]
        if not eval_pocket_ids:
            eval_pocket_ids = shuffled[-1:]
        candidate_ligand_ids = ligand_ids

        for pid in train_pockets:
            for lid, label in bundle.labels.get(pid, {}).items():
                train_pairs.append((pid, lid, label))
        notes = "target split by held-out pockets"
    elif split_name == "standard":
        # Standard split: random pair split while keeping all entities available at eval.
        eval_pocket_ids = pocket_ids
        candidate_ligand_ids = ligand_ids
        all_pairs: List[Tuple[str, str, int]] = []
        for pid in pocket_ids:
            for lid, label in bundle.labels.get(pid, {}).items():
                all_pairs.append((pid, lid, label))
        rng.shuffle(all_pairs)
        cut = max(1, int(0.8 * len(all_pairs)))
        train_pairs = all_pairs[:cut]
        notes = "standard random pair split"
    elif split_name == "scaffold":
        # Scaffold split: hold out ligand scaffolds.
        scaffolds = sorted({l.scaffold for l in bundle.ligands})
        rng.shuffle(scaffolds)
        cut = max(1, int(0.7 * len(scaffolds)))
        train_scafs = set(scaffolds[:cut])
        test_scafs = set(scaffolds[cut:])
        if not test_scafs:
            test_scafs = {scaffolds[-1]}

        train_lids = [l.lid for l in bundle.ligands if l.scaffold in train_scafs]
        candidate_ligand_ids = [l.lid for l in bundle.ligands if l.scaffold in test_scafs]
        eval_pocket_ids = pocket_ids

        for pid in pocket_ids:
            for lid in train_lids:
                if lid in bundle.labels.get(pid, {}):
                    train_pairs.append((pid, lid, bundle.labels[pid][lid]))
        notes = "scaffold split by held-out ligand scaffolds"
    else:
        # Joint OOD split: hold out both pockets and ligand scaffolds.
        shuffled = pocket_ids[:]
        rng.shuffle(shuffled)
        cut_p = max(1, int(0.8 * len(shuffled)))
        train_pockets = set(shuffled[:cut_p])
        eval_pocket_ids = shuffled[cut_p:]
        if not eval_pocket_ids:
            eval_pocket_ids = shuffled[-1:]

        scaffolds = sorted({l.scaffold for l in bundle.ligands})
        rng.shuffle(scaffolds)
        cut_s = max(1, int(0.7 * len(scaffolds)))
        train_scafs = set(scaffolds[:cut_s])
        eval_scafs = set(scaffolds[cut_s:])
        if not eval_scafs:
            eval_scafs = {scaffolds[-1]}

        train_lids = {l.lid for l in bundle.ligands if l.scaffold in train_scafs}
        candidate_ligand_ids = [l.lid for l in bundle.ligands if l.scaffold in eval_scafs]
        if not candidate_ligand_ids:
            candidate_ligand_ids = [l.lid for l in bundle.ligands]

        for pid in train_pockets:
            for lid in train_lids:
                if lid in bundle.labels.get(pid, {}):
                    train_pairs.append((pid, lid, bundle.labels[pid][lid]))
        notes = "joint_ood split by held-out pockets and held-out ligand scaffolds"

    return SplitView(
        split_name=split_name,
        train_pairs=train_pairs,
        eval_pocket_ids=eval_pocket_ids,
        candidate_ligand_ids=candidate_ligand_ids,
        notes=notes,
    )
