# Base Siege for Gymnasium

A single-agent strategy environment: choose an army within a resource budget,
then deploy it around a defended base. The simulator controls movement and combat.
Public package author: **Vaibhav Bairathi**, Master of Artificial Intelligence,
University of Technology Sydney. Derived from the Base Siege capstone environment;
the upstream source and revision are recorded in `PROVENANCE.json`.

**License: MIT.** See `LICENSE`. Copyright (c) 2026 Vaibhav Bairathi.

**Release candidate: not yet published to PyPI.** Upload and installation
verification are still required. This package is maintained independently of
Gymnasium/Farama; registration does not imply their endorsement.

## Install this source package

Use Python 3.12. Open a terminal in this folder (the folder containing
`pyproject.toml`), then run:

```bash
python -m pip install -e ".[render,dev]"
python examples/random_agent.py
```

For simulation without rendering, only Gymnasium and NumPy are required:
`python -m pip install .`

## Run one attack

Save this as `play.py`, then run `python play.py` in the same terminal:

```python
import gymnasium as gym
import gymnasium_base_siege  # registers BaseSiege-v0

env = gym.make("BaseSiege-v0", base_seed=42)
obs, info = env.reset(seed=0)
env.action_space.seed(7)

while True:
    action = env.action_space.sample(mask=info["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Destruction:", 100 * info["destruction"], "%")
        break
env.close()
```

Always use the legal-action mask. An illegal action raises `ValueError` before
changing the state. A terminal mask is all zeros: reset instead of sampling again.
`env.unwrapped.action_masks()` also returns a Boolean mask for integrations that
expect that method. Standard algorithms do not automatically use masks.

## Change troop and defence settings

Pass only the values you want to change. Every other value keeps its original default.
Settings belong to each environment; no global config tables are modified.

```python
env = gym.make(
    "BaseSiege-v0",
    troop_stats={
        "tank": {"hp": 1200, "speed": 0.20, "damage_per_tick": 12},
        "ranged": {"hp": 110, "damage_per_tick": 18, "range": 5},
    },
    defence_stats={
        "cannon": {"hp": 800, "damage_per_shot": 8, "range": 4},
        "mortar": {"damage_per_shot": 60, "attack_cooldown": 6},
    },
)
```

For example, the default tank moves at 0.10 tiles per tick. Setting its speed to
0.20 makes it accumulate one tile of movement in five ticks instead of ten.
Actual progress still depends on targets, paths and combat. Defences do not move:
`attack_cooldown` controls firing frequency. A cooldown of 6 means six ticks
between shots; larger values mean slower firing.

| Group | Unit names | Allowed settings |
|---|---|---|
| `troop_stats` | `tank`, `melee`, `ranged`, `breaker` | `hp`, `damage_per_tick`, `speed`, `range`, `cost` |
| `defence_stats` | `cannon`, `mortar` | `hp`, `damage_per_shot`, `range`, `min_range`, `attack_cooldown`, `splash_radius` |

HP, cost and cooldown must be positive integers. Damage must be a non-negative
integer. Troop speed must be between 0.01 and 1.00 tiles per tick, in steps of
0.01: the simulator uses integer movement credits and allows at most one tile
of movement per tick. Troop range must be at least 1. Defence range, minimum range
and splash radius must be non-negative; minimum range cannot exceed range.
Unknown names and invalid values are rejected, not silently ignored.

Create a new environment to use new settings; do not mutate internal state during
an episode. Budget remains 200 and the army cap remains 25 even with custom costs.
With `fixed_army=True`, custom costs that put the fixed army over budget are rejected.

Save the resolved settings alongside results:

```python
import json
from pathlib import Path

Path("settings.json").write_text(json.dumps(env.unwrapped.settings, indent=2))
saved = json.loads(Path("settings.json").read_text())
another_env = gym.make("BaseSiege-v0", **saved)
```

`info["settings_hash"]` identifies the stat settings. It does not identify a layout,
seed, software version or complete experiment; record those separately. Custom
settings create a different task and are not directly comparable to the frozen
capstone results. Stats are not added to the observation: agents do not receive an
explicit description of customized damage, speed or HP limits.

## Environment contract

| Property | Default |
|---|---|
| Gymnasium ID | `BaseSiege-v0` |
| Board | 16 × 16 |
| Resource budget / army cap | 200 / 25 |
| Actions | `Discrete(246)`; masked by phase, budget and reserve |
| Grid observation | float32, `(8,16,16)`, values in `[0,1]` |
| Vector observation | float32, `(9,)`, values in `[0,1]` |
| Battle limit | 300 simulation ticks, terminal |
| Safety guard | 70 agent decisions, truncation |
| Rendering | `rgb_array`, optional Matplotlib dependency |

The agent buys one troop per action, sees the next observation, and can buy again
or finish shopping. Shopping and finishing shopping do not advance combat time.
During battle, deploying one troop or waiting advances up to ten simulator ticks.
A Gymnasium step is one decision, not always ten ticks.

Actions 0–3 buy tank/melee/ranged/breaker; 4 finishes shopping; 5–244 deploy one
troop at one of 60 perimeter tiles; 245 waits. Deployment uses
`5 + troop_index * 60 + tile_index`. `deployment_action("ranged", 0)` and
`describe_action(action)` are exported helpers. Perimeter positions run clockwise
from the top-left corner.

The eight grid channels are Town Hall, cannon, mortar, storage, wall, building
health fraction, defence coverage divided by five, and troop density divided by
eight (clipped at one). The vector contains four reserve counts divided by 25,
remaining budget / 200, remaining battle time / 300, phase (0 shopping, 1 battle),
destroyed building-value fraction and live troop count / 25.

Reward combines newly destroyed building value divided by initial building value,
three one-off bonuses of 0.25 (half destruction, Town Hall destroyed, full clear),
and one cost penalty of `0.30 * spent / 200`. Walls score zero; wall shaping is off.
Destruction measures **building value**, not number of buildings or win rate.

An attack terminates when every scoring building is destroyed, there are no live
or reserve troops, or 300 ticks have elapsed. The separate safety guard truncates.

## Layouts, rendering and limitations

Default motifs are `closed_ring`, `ns_gates`, and `ew_gates`. To include the fourth:
`gym.make("BaseSiege-v0", motifs=("layered_core",))`.
`reset(seed=...)` controls layout sampling. `base_seed=42` fixes the layout seed;
fix a single motif as well to fix both layout inputs. `BaseSpec`, `generate_base`
and `reset(options={"spec": spec})` support explicit layouts. User-created specs
must satisfy the game geometry; the public API is not an untrusted-layout validator.

This package generates layouts directly. It does not bundle the capstone's
filtered pools, checkpoints, learning algorithms, training logs or web interface.
The original seed guard remains: layout seeds 11000–11999 require explicit
`allow_final=True`; ordinary generated training seeds are 0–9999.

Use `render_mode="rgb_array"`, call `reset()`, then `frame = env.render()` to obtain
an RGB NumPy image. For headless machines set the Matplotlib backend to `Agg`.

Movement uses four-neighbour A*; attack range uses Euclidean distance. Walls do
not block shots. The known inaccessible-target movement limitation remains:
units may keep an unreachable preferred target. No reachable-target fallback
has been added. Observations omit individual troop health, type and target,
so the observation is partially observable. Manual play in the separate web
application uses different deployment timing.

The default Gymnasium passive checks remain enabled. Gymnasium's standalone
`check_env` samples unmasked actions, so it can raise our intentional illegal-action
error. Use the included mask-aware contract tests; do not interpret this as a
claim that an unmodified `check_env` run passed.

## Validation and publication

From this package folder, after installing the development extra:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

See `docs/PUBLISHING.md` for the release steps, `docs/VALIDATION.md` for measured
checks, and `PROVENANCE.json` for the upstream commit and source hashes.
The original research repository is not modified by this extraction.
