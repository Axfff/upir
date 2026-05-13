import hashlib
import random
import re
from pathlib import Path
from typing import List, Sequence, Tuple


def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def l2_norm(v: Sequence[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def unit(v: Sequence[float]) -> List[float]:
    n = max(1e-12, l2_norm(v))
    return [x / n for x in v]


def embed_token(token: str, dim: int, base_seed: int) -> List[float]:
    rng = random.Random(stable_seed(f"{base_seed}:{token}"))
    return unit([rng.gauss(0.0, 1.0) for _ in range(dim)])


def kmer_windows(text: str, k: int) -> List[str]:
    s = (text or "").strip()
    if not s:
        return ["UNK"]
    if len(s) <= k:
        return [s]
    return [s[i : i + k] for i in range(len(s) - k + 1)]


def text_to_tokens(text: str, dim: int, k: int, max_tokens: int, base_seed: int) -> List[List[float]]:
    windows = kmer_windows(text, k)
    if len(windows) > max_tokens:
        stride = len(windows) / max_tokens
        windows = [windows[int(i * stride)] for i in range(max_tokens)]
    return [embed_token(w, dim, base_seed) for w in windows]


def parse_smi(path: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                smi = parts[0]
                cid = str(stable_seed(smi))
            else:
                smi = parts[0]
                cid = parts[-1]
            items.append((smi, cid))
    return items


def parse_residue_sequence_from_mol2(path: Path, max_residues: int) -> str:
    residues: List[str] = []
    seen = set()
    in_atom = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom:
                break
            if not in_atom:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            resid = parts[7]
            aa = re.sub(r"[^A-Z]", "", resid.upper())
            if not aa:
                continue
            if resid in seen:
                continue
            seen.add(resid)
            residues.append(aa)
            if len(residues) >= max_residues:
                break
    return " ".join(residues)


def scaffold_from_smiles(smi: str) -> str:
    core = re.sub(r"[^A-Za-z0-9]", "", smi)
    core = re.sub(r"\d", "", core)
    core = core[:16] if core else "UNK"
    return f"S::{core}"
