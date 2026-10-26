"""Accidental-access guards for the reserved evaluation partition."""
from numbers import Integral
from gymnasium_base_siege.core import config as C


def ensure_seed_allowed(seed: int, *, allow_final: bool = False) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Base seed must be a non-negative integer")
    seed = int(seed)
    if C.FINAL_SEED_RANGE[0] <= seed <= C.FINAL_SEED_RANGE[1] and not allow_final:
        raise PermissionError("Final seeds are reserved until protocol freeze")
    return seed
