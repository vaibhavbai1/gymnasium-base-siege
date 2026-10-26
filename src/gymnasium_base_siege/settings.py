"""Validated, instance-local overrides for the public environment."""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real

from .core import config as C

TROOP_FIELDS = frozenset({"hp", "damage_per_tick", "speed", "range", "cost"})
DEFENCE_FIELDS = frozenset({"hp", "damage_per_shot", "range", "min_range",
                            "attack_cooldown", "splash_radius"})
INTEGER_FIELDS = frozenset({"hp", "damage_per_tick", "damage_per_shot", "cost", "attack_cooldown"})


def _merge(defaults, overrides, allowed_types, fields):
    result = deepcopy(defaults)
    if overrides is None:
        return result
    if not isinstance(overrides, Mapping):
        raise TypeError("Settings must be a dictionary of unit names and stat dictionaries")
    for kind, changes in overrides.items():
        if kind not in allowed_types:
            raise ValueError(f"Unknown unit {kind!r}; choose from {tuple(allowed_types)}")
        if not isinstance(changes, Mapping):
            raise TypeError(f"Settings for {kind} must be a dictionary")
        for key, value in changes.items():
            if key not in fields:
                raise ValueError(f"Unsupported setting {kind}.{key}; choose from {sorted(fields)}")
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{kind}.{key} must be a finite number")
            if key in INTEGER_FIELDS and not isinstance(value, Integral):
                raise ValueError(f"{kind}.{key} must be an integer")
            lower = 1 if key in {"hp", "cost", "attack_cooldown"} else 0
            if value < lower:
                raise ValueError(f"{kind}.{key} must be at least {lower}")
            if key == "speed" and (not 0 < value <= 1 or
                not math.isclose(value * C.SPEED_SCALE, round(value * C.SPEED_SCALE), abs_tol=1e-9, rel_tol=0)):
                raise ValueError("Troop speed must be 0.01 to 1.00 tiles/tick, in steps of 0.01")
            if kind in C.TROOP_TYPES and key == "range" and value < 1:
                raise ValueError("Troop range must be at least 1 tile")
            result[kind][key] = int(value) if key in INTEGER_FIELDS else float(value)
        if kind in C.DEFENCE_TYPES and result[kind]["min_range"] > result[kind]["range"]:
            raise ValueError(f"{kind}.min_range cannot exceed its range")
    return result


def resolve_settings(troop_stats=None, defence_stats=None):
    troops = _merge(C.TROOP_STATS, troop_stats, C.TROOP_TYPES, TROOP_FIELDS)
    buildings = _merge(C.BUILDING_STATS, defence_stats, C.DEFENCE_TYPES, DEFENCE_FIELDS)
    # Canonical numeric types make saved/reloaded settings hashes identical.
    for table, fields in ((troops, TROOP_FIELDS), (buildings, DEFENCE_FIELDS)):
        for stats in table.values():
            for key in fields:
                stats[key] = int(stats[key]) if key in INTEGER_FIELDS else float(stats[key])
    return troops, buildings
