"""Seeded RNG utilities — the L1.1 contract.

Lesson L1.1 (docs/reference/lessons_learned.md): unseeded resets and unseeded
trainer RNG were the single largest reproducibility debt of paper 1. The pe2
contract, enforced from this commit:

1. ONE root seed per run (from a frozen config).
2. All randomness flows through `np.random.Generator` instances created here.
   No `np.random.*` global calls, no `random` module, no unseeded
   `default_rng()` anywhere else in pe2.
3. Independent sub-streams (env resets, eval init sets, trainer exploration,
   opponent sampling, ...) are derived from the root seed by NAME, so adding
   a new stream never shifts the draws of existing streams.
"""

import hashlib

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create the canonical Generator for a raw seed."""
    return np.random.default_rng(seed)


def _label_to_int(label: str) -> int:
    """Stable label hash (sha256 — Python's hash() is salted per-process)."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def derive_seed(root_seed: int, label: str) -> int:
    """Derive an independent child seed for a named sub-stream.

    Uses SeedSequence(entropy=root_seed, spawn_key=(label_int,)): the label
    sits in the spawn key, so streams are addressed by NAME and inserting a
    new stream never shifts existing ones.
    """
    seq = np.random.SeedSequence(entropy=root_seed, spawn_key=(_label_to_int(label),))
    return int(seq.generate_state(1, dtype=np.uint64)[0])
