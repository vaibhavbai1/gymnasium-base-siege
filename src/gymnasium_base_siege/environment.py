"""Public API with validated settings and legal-action masks."""
from copy import deepcopy
import hashlib
import json
import gymnasium as gym

from .core.siege_env import SiegeEnv
from .core import config as C
from .settings import resolve_settings, TROOP_FIELDS, DEFENCE_FIELDS


class BaseSiegeEnv(SiegeEnv):
    """Base Siege with masks in info and Gymnasium render-mode handling.

    Illegal actions raise ValueError before simulation state changes. Always
    use info['action_mask'] or action_masks() when choosing an action.
    """

    def __init__(self, *, troop_stats=None, defence_stats=None, **kwargs):
        if "building_stats" in kwargs:
            raise TypeError("Use defence_stats for cannon and mortar overrides")
        troops, buildings = resolve_settings(troop_stats, defence_stats)
        if kwargs.get("fixed_army", False):
            cost = sum(troops[t]["cost"] * n for t, n in C.HEURISTIC_ARMY.items())
            if cost > C.RESOURCE_BUDGET:
                raise ValueError("Custom costs make the fixed army exceed the 200-resource budget")
        super().__init__(troop_stats=troops, building_stats=buildings, **kwargs)
        self.settings_hash = hashlib.sha256(
            json.dumps(self.settings, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def settings(self):
        """Return an independent, JSON-serializable copy for experiment records."""
        return {"troop_stats": {k: {f: deepcopy(v[f]) for f in sorted(TROOP_FIELDS)}
                                for k, v in self._troop_stats.items()},
                "defence_stats": {k: {f: deepcopy(self._building_stats[k][f]) for f in sorted(DEFENCE_FIELDS)}
                                  for k in C.DEFENCE_TYPES}}

    def set_fixed_army(self, value):
        if value and sum(self._troop_stats[t]["cost"] * n for t, n in C.HEURISTIC_ARMY.items()) > C.RESOURCE_BUDGET:
            raise ValueError("Custom costs make the fixed army exceed the 200-resource budget")
        super().set_fixed_army(value)

    def _info(self):
        return {**super()._info(), "action_mask": self.action_masks().astype("int8"),
                "settings_hash": self.settings_hash}

    def snapshot(self):
        # Original snapshots are JSON serializable; keep that property after
        # adding the NumPy mask to the public reset/step info dictionaries.
        result = super().snapshot()
        result["action_mask"] = result["action_mask"].tolist()
        for building in result["buildings"]:
            stats = self._building_stats[building["type"]]
            building["range"] = stats["range"]
            building["min_range"] = stats["min_range"]
        return result

    def render(self):
        if self.render_mode is None:
            return None
        if self.state is None:
            raise gym.error.ResetNeeded("Call reset before render")
        try:
            return super().render()
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("matplotlib"):
                raise gym.error.DependencyNotInstalled(
                    'Install rendering with: pip install "gymnasium-base-siege[render]"'
                ) from exc
            raise
