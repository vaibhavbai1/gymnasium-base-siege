"""Gymnasium interface shared by manual play, baselines and learned policies."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core.access import ensure_seed_allowed
from gymnasium_base_siege.core.combat import BattleState, deploy_troop, record_snapshot, step_tick, _step_7_check_end
from gymnasium_base_siege.core.generator import BaseSpec, generate_base


def deployment_action(troop_type: str, tile_index: int) -> int:
    if not 0 <= tile_index < C.N_DEPLOY_TILES:
        raise ValueError("Deployment index outside the perimeter")
    return C.ACTION_DEPLOY_FIRST + C.TROOP_TYPES.index(troop_type) * C.N_DEPLOY_TILES + tile_index


def describe_action(action: int) -> str:
    if 0 <= action < 4:
        return f"Buy {C.TROOP_TYPES[action]}"
    if action == C.ACTION_FINISH_SHOPPING:
        return "Finish shopping"
    if C.ACTION_DEPLOY_FIRST <= action <= C.ACTION_DEPLOY_LAST:
        kind, index = divmod(action - C.ACTION_DEPLOY_FIRST, C.N_DEPLOY_TILES)
        return f"Deploy {C.TROOP_TYPES[kind]} at {C.DEPLOY_TILES[index]}"
    if action == C.ACTION_WAIT:
        return "Wait 10 ticks"
    raise ValueError("Unknown action")


class SiegeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, *, pool: list[BaseSpec] | None = None,
                 motifs=C.TRAIN_MOTIFS, base_seed: int | None = None,
                 fixed_army: bool = False, render_mode=None,
                 allow_final=False, record=False, record_ticks=False,
                 troop_stats=None, building_stats=None):
        super().__init__()
        if not motifs or any(m not in C.MOTIFS for m in motifs):
            raise ValueError("Choose at least one known motif")
        if pool is not None and not pool:
            raise ValueError("An empty pool cannot be sampled")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        self.pool, self.motifs = pool, tuple(motifs)
        self.base_seed, self.fixed_army = base_seed, fixed_army
        self.allow_final, self.record, self.render_mode = allow_final, record, render_mode
        # Optional presentation recording only; never enabled by the trainer.
        self.record_ticks = record_ticks
        if base_seed is not None:
            ensure_seed_allowed(base_seed, allow_final=allow_final)
        for spec in pool or []:
            ensure_seed_allowed(spec.seed, allow_final=allow_final)
        self.action_space = spaces.Discrete(C.N_ACTIONS)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, (C.N_GRID_LAYERS, C.GRID_SIZE, C.GRID_SIZE), np.float32),
            "vector": spaces.Box(0, 1, (C.VECTOR_DIM,), np.float32),
        })
        self._troop_stats = C.TROOP_STATS if troop_stats is None else troop_stats
        self._building_stats = C.BUILDING_STATS if building_stats is None else building_stats
        self.state = None
        self._coords = np.indices((C.GRID_SIZE, C.GRID_SIZE))

    def set_fixed_army(self, value: bool):
        self.fixed_army = bool(value)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "spec" in options:
            self.base_spec = options["spec"]
            if not isinstance(self.base_spec, BaseSpec):
                raise TypeError("spec must be a BaseSpec")
        elif self.pool is not None and "base_seed" not in options:
            self.base_spec = self.pool[int(self.np_random.integers(len(self.pool)))]
        else:
            layout_seed = options.get("base_seed", self.base_seed)
            if layout_seed is None:
                layout_seed = int(self.np_random.integers(C.TRAIN_SEED_RANGE[1] + 1))
            motif = options.get("motif", self.motifs[int(self.np_random.integers(len(self.motifs)))])
            if motif not in self.motifs:
                raise ValueError("Motif is outside this environment's configured set")
            self.base_spec = generate_base(layout_seed, motif, allow_final=self.allow_final)
        ensure_seed_allowed(self.base_spec.seed, allow_final=self.allow_final)
        army = C.HEURISTIC_ARMY if self.fixed_army else None
        self.state = BattleState.from_spec(self.base_spec, reserve=army,
            troop_stats=self._troop_stats, building_stats=self._building_stats)
        self.state.phase = "battle" if self.fixed_army else "shopping"
        self.decisions, self.episode_return, self.truncated = 0, 0.0, False
        self._cost_pending = self.fixed_army
        self.reward_totals = dict.fromkeys(C.REWARD_COMPONENT_KEYS, 0.0)
        self.last_components = dict(self.reward_totals)
        self.events, self.history = [], []
        self.visual_history = []
        self.last_action = None
        self._record()
        return self._observation(), self._info()

    def action_masks(self):
        mask = np.zeros(C.N_ACTIONS, dtype=bool)
        if self.state is None or self.state.is_finished or self.truncated:
            return mask
        s = self.state
        if s.phase == "shopping":
            if sum(s.reserve.values()) < C.MAX_ARMY_SIZE:
                mask[:4] = [s.spent + self._troop_stats[t]["cost"] <= C.RESOURCE_BUDGET for t in C.TROOP_TYPES]
            mask[C.ACTION_FINISH_SHOPPING] = True
        else:
            for kind, troop in enumerate(C.TROOP_TYPES):
                start = C.ACTION_DEPLOY_FIRST + kind * C.N_DEPLOY_TILES
                mask[start:start + C.N_DEPLOY_TILES] = s.reserve.get(troop, 0) > 0
            mask[C.ACTION_WAIT] = True
        return mask

    def step(self, action):
        if self.state is None:
            raise gym.error.ResetNeeded("Call reset before step")
        if self.state.is_finished or self.truncated:
            raise gym.error.ResetNeeded("The episode has ended")
        if not self.action_space.contains(action) or not self.action_masks()[int(action)]:
            raise ValueError(f"Illegal action: {action}")
        action = int(action)
        s = self.state
        previous_value = s.destroyed_value
        previous_milestones = dict(s.milestones)
        previous_walls = sum(b.is_wall for b in s.buildings)
        components = dict.fromkeys(C.REWARD_COMPONENT_KEYS, 0.0)
        self.events, self.last_action = [], action
        if s.phase == "shopping":
            if action == C.ACTION_FINISH_SHOPPING:
                s.phase = "battle"
                self._cost_pending = True
                _step_7_check_end(s)
            else:
                troop = C.TROOP_TYPES[action]
                s.reserve[troop] = s.reserve.get(troop, 0) + 1
                s.spent += self._troop_stats[troop]["cost"]
        else:
            if action != C.ACTION_WAIT:
                kind, index = divmod(action - C.ACTION_DEPLOY_FIRST, C.N_DEPLOY_TILES)
                deploy_troop(s, C.TROOP_TYPES[kind], C.DEPLOY_TILES[index])
                self._record_tick(clear_attacks=True)
            for _ in range(C.TICKS_PER_DECISION):
                if s.is_finished:
                    break
                step_tick(s)
                self._record_tick()
                self.events.extend({**attack, "tick": s.tick} for attack in s.last_attacks)
        if self._cost_pending:
            components["cost_penalty"] = -C.COST_PENALTY_WEIGHT * s.spent / C.RESOURCE_BUDGET
            self._cost_pending = False
        components["destruction"] = (s.destroyed_value - previous_value) / max(1, s.total_base_value)
        bonuses = {"half": C.MILESTONE_HALF_DESTRUCTION,
                   "town_hall": C.MILESTONE_TOWN_HALL, "full_clear": C.MILESTONE_FULL_CLEAR}
        components["milestone"] = sum(v for k, v in bonuses.items() if s.milestones[k] and not previous_milestones[k])
        components["wall_shaping"] = C.WALL_SHAPING_BONUS * (previous_walls - sum(b.is_wall for b in s.buildings))
        reward = float(sum(components.values()))
        self.decisions += 1
        self.truncated = not s.is_finished and self.decisions >= C.MAX_DECISIONS_PER_EPISODE
        self.episode_return += reward
        self.last_components = components
        for key, value in components.items():
            self.reward_totals[key] += value
        self._record()
        return self._observation(), reward, s.is_finished, self.truncated, self._info()

    def _observation(self):
        grid = np.zeros(self.observation_space["grid"].shape, dtype=np.float32)
        s = self.state
        for building in s.buildings:
            r, c = building.position
            grid[C.BUILDING_TYPES.index(building.building_type), r, c] = 1
            grid[C.GRID_LAYER_BUILDING_HEALTH, r, c] = building.health_fraction
            if building.is_defence:
                distance = np.hypot(self._coords[0] - r, self._coords[1] - c)
                grid[C.GRID_LAYER_DEFENCE_COVERAGE] += ((distance >= building.min_range) &
                    (distance <= building.attack_range)).astype(np.float32) / C.MAX_DEFENCE_COVERAGE
        for troop in s.troops:
            grid[C.GRID_LAYER_TROOP_DENSITY, *troop.position] += 1 / C.MAX_TROOP_DENSITY_PER_TILE
        np.clip(grid, 0, 1, out=grid)
        vector = np.array([*[s.reserve.get(t, 0) / C.MAX_ARMY_SIZE for t in C.TROOP_TYPES],
            (C.RESOURCE_BUDGET - s.spent) / C.RESOURCE_BUDGET,
            max(0, C.BATTLE_TICK_LIMIT - s.tick) / C.BATTLE_TICK_LIMIT,
            float(s.phase == "battle"), s.destroyed_fraction,
            s.troops_alive / C.MAX_ARMY_SIZE], dtype=np.float32)
        return {"grid": grid, "vector": vector}

    def _info(self):
        return {"base_seed": self.base_spec.seed, "motif": self.base_spec.motif,
                "destruction": self.state.destroyed_fraction,
                "town_hall": self.state.milestones["town_hall"],
                "full_clear": self.state.milestones["full_clear"],
                "finish_reason": "safety_guard" if self.truncated else self.state.finish_reason,
                "tick": self.state.tick, "decisions": self.decisions,
                "spent": self.state.spent, "reward_components": dict(self.last_components),
                "reward_totals": dict(self.reward_totals), "episode_return": self.episode_return}

    def snapshot(self):
        return {**record_snapshot(self.state), **self._info(),
                "legal_actions": np.flatnonzero(self.action_masks()).tolist(),
                "last_action": self.last_action, "events": list(self.events)}

    def _record(self):
        if self.record:
            self.history.append(self.snapshot())
        if self.record_ticks:
            frame = self.snapshot()
            self._add_visual_movement(frame)
            self.visual_history.append(frame)

    def _add_visual_movement(self, frame):
        credits = {t.entity_id: t.movement_fraction for t in self.state.troops}
        for troop in frame['troops']:
            troop['movement_fraction'] = credits[troop['entity_id']]

    def _record_tick(self, clear_attacks=False):
        if self.record_ticks:
            # No mutation or extra simulation: capture the tick just executed.
            frame = record_snapshot(self.state)
            frame.update(destruction=self.state.destroyed_fraction,
                         town_hall=self.state.milestones['town_hall'],
                         full_clear=self.state.milestones['full_clear'])
            if clear_attacks:
                frame['attacks'] = []
            self._add_visual_movement(frame)
            self.visual_history.append(frame)

    def render(self):
        from gymnasium_base_siege.core.render import draw_snapshot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        draw_snapshot(self.snapshot(), ax, style="presentation")
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return pixels
