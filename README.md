# Base Siege for Gymnasium

[![PyPI version](https://img.shields.io/pypi/v/gymnasium-base-siege)](https://pypi.org/project/gymnasium-base-siege/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://pypi.org/project/gymnasium-base-siege/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**A customizable, single-agent strategy environment for reinforcement learning.**

An agent buys troops within a resource budget, chooses where and when to deploy them, and attempts to destroy a defended base. The simulator handles troop movement, targeting and combat. Users can change troop and defence statistics without editing the simulator.

**Public package author:** Vaibhav Bairathi · Master of Artificial Intelligence, University of Technology Sydney.

**Published release:** [`0.1.0` on PyPI](https://pypi.org/project/gymnasium-base-siege/0.1.0/) · **Environment ID:** `BaseSiege-v0` · **License:** MIT.

[Source code](https://github.com/vaibhavbai1/gymnasium-base-siege) · [Report an issue](https://github.com/vaibhavbai1/gymnasium-base-siege/issues) · [Related capstone web demo](https://base-siege-dihedrals.streamlit.app/)

This is an independently maintained environment using the Gymnasium API. It is not an official Farama-maintained environment or an endorsed benchmark.

## Features

- A 16 × 16 grid with procedural base layouts.
- Two decision phases: army purchasing, followed by deployment and waiting.
- Four troop types, two defence types and 60 perimeter deployment positions.
- A discrete action space with legal-action masks.
- Structured observations containing a spatial grid and summary vector.
- Separate troop and defence settings for each environment instance.
- Seeded layout generation and reproducible action sampling when explicitly seeded.
- Optional RGB rendering and JSON-compatible snapshots.

## Installation

**Version 0.1.0 requires Python 3.12.** Its declared Python range is `>=3.12,<3.13`.

Run these commands in a terminal using your chosen Python 3.12 environment. A repository clone is not needed for installation from PyPI.

```bash
python -m pip install gymnasium-base-siege
```

For rendering support:

```bash
python -m pip install "gymnasium-base-siege[render]"
```

For an experiment that must use this exact release:

```bash
python -m pip install "gymnasium-base-siege==0.1.0"
```

On Windows, if `python` points to a different version, use `py -3.12` instead. The base installation requires Gymnasium and NumPy. Matplotlib is optional; PyTorch and Stable-Baselines3 are not required to run the simulator.

## Quick start: play one attack

Save the following as `play.py`. In a terminal opened in the folder containing that file, run `python play.py` using the environment where you installed the package.

```python
import gymnasium as gym
import gymnasium_base_siege  # Registers BaseSiege-v0.

env = gym.make(
    "BaseSiege-v0",
    base_seed=42,
    motifs=("closed_ring",),
)
env.action_space.seed(123)
observation, info = env.reset(seed=0)

try:
    while True:
        action = env.action_space.sample(mask=info["action_mask"])
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Destruction: {100 * info['destruction']:.2f}%")
            print(f"Episode return: {info['episode_return']:.4f}")
            print(f"Decisions: {info['decisions']}")
            print(f"Finish reason: {info['finish_reason']}")
            break
finally:
    env.close()
```

This example uses a **random legal-action policy**, not a trained model. It can make poor purchases or finish shopping without buying troops. Its score is only a demonstration that the environment runs.

Importing `gymnasium_base_siege` registers the environment. Gymnasium also supports the module-qualified form:

```python
import gymnasium as gym

env = gym.make("gymnasium_base_siege:BaseSiege-v0")
observation, info = env.reset(seed=0)
env.close()
```

## Legal-action masks

Not every action is available in every state. For example, deploying is illegal during shopping, and buying a troop is illegal when there is not enough budget.

| Access | Returned value |
|---|---|
| `info["action_mask"]` after `reset()` or `step()` | An `int8` array of length 246; 1 means legal |
| `env.unwrapped.action_masks()` | A Boolean array of length 246 |

Use the latest mask when selecting an action. An illegal action raises `ValueError` before changing the simulation state. When an episode ends, its mask is all zeros: call `reset()` before selecting another action.

A mask must be incorporated into the chosen learning algorithm's action selection and, where applicable, its training update. Installing this environment does not make an ordinary PPO, DQN or other implementation automatically support masking. This package supplies the environment, not training algorithms or trained checkpoints.

## Customize troop and defence settings

Pass only the values you want to change. Unspecified values retain their defaults. Each environment owns its settings; changing one does not modify another environment or the module's default tables.

```python
import gymnasium as gym
import gymnasium_base_siege

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
observation, info = env.reset(seed=42)
print(env.unwrapped.settings)
env.close()
```

For example, changing tank speed from `0.10` to `0.20` makes it accumulate one tile of movement in five ticks instead of ten. Actual progress also depends on paths, targets and combat. Defences are stationary: their firing frequency is controlled by `attack_cooldown`. A value of `6` means six ticks between shots when a valid target remains available.

### Supported settings

| Group | Unit names | Settings |
|---|---|---|
| `troop_stats` | `tank`, `melee`, `ranged`, `breaker` | `hp`, `damage_per_tick`, `speed`, `range`, `cost` |
| `defence_stats` | `cannon`, `mortar` | `hp`, `damage_per_shot`, `range`, `min_range`, `attack_cooldown`, `splash_radius` |

Validation rules:

- HP, cost and attack cooldown must be positive integers.
- Damage must be a non-negative integer.
- Troop speed must be between `0.01` and `1.00` tiles per tick, in increments of `0.01`. Movement uses integer credits and supports at most one tile per tick.
- Troop attack range must be at least one tile.
- Defence range, minimum range and splash radius must be non-negative. Minimum range cannot exceed maximum range.
- Values must be finite. Unknown unit names and unsupported setting names are rejected.

Create a new environment to change settings. The public API does not support changing them during an episode. The resource budget remains 200 and the army cap remains 25, including when troop costs are customized.

### Default troop statistics

| Troop | HP | Damage per attack tick | Speed: tiles/tick | Range: tiles | Cost |
|---|---:|---:|---:|---:|---:|
| Tank | 900 | 8 | 0.10 | 1 | 25 |
| Melee | 120 | 15 | 0.16 | 1 | 8 |
| Ranged | 90 | 12 | 0.14 | 4 | 10 |
| Breaker | 60 | 350 | 0.22 | 1 | 12 |

Tanks prefer defences and then fall back to other scoring buildings when no defences remain. Melee and ranged troops target scoring buildings. Breakers target walls only and are consumed after their strike. Troops do not deal damage on ticks when they are moving instead of attacking.

### Default defence statistics

| Defence | HP | Damage per shot | Range: tiles | Minimum range | Cooldown: ticks | Splash radius |
|---|---:|---:|---:|---:|---:|---:|
| Cannon | 600 | 5 | 3 | 0 | 1 | 0 |
| Mortar | 500 | 45 | 5 | 2 | 5 | 1.5 |

The mortar has a blind spot and splash damage. Damage outside its impact tile uses the simulator's fixed splash falloff. Wall, Town Hall and storage statistics are not exposed as constructor overrides in this release.

### Save and reuse settings

This complete example writes a JSON file in the script's working directory and recreates the same stat settings:

```python
import json
from pathlib import Path
import gymnasium as gym
import gymnasium_base_siege

env = gym.make("BaseSiege-v0", troop_stats={"tank": {"hp": 1200}})
settings_path = Path("settings.json")
settings_path.write_text(
    json.dumps(env.unwrapped.settings, indent=2), encoding="utf-8"
)

saved = json.loads(settings_path.read_text(encoding="utf-8"))
restored_env = gym.make("BaseSiege-v0", **saved)
assert restored_env.unwrapped.settings == env.unwrapped.settings

env.close()
restored_env.close()
```

`settings` returns a copy. `info["settings_hash"]` identifies the resolved troop and defence settings. It does not identify the layout, seed, package version or complete experiment; record those separately.

Customized statistics define a different task from the original capstone benchmark. The observation does not explicitly encode all customized values, such as damage or movement speed, so an agent is not automatically told which settings were changed.

## How an episode works

1. **Reset:** create or select a base, initialize the budget and enter shopping.
2. **Shopping:** buy one troop per action, observe the updated state, and decide whether to buy again or finish shopping.
3. **Battle:** deploy one reserve troop at a perimeter position, or wait. Movement, targeting and combat proceed automatically.
4. **Finish:** return the terminal or truncation flag and the final metrics.

| Property | Default |
|---|---|
| Board | 16 × 16 tiles |
| Resource budget | 200 |
| Maximum army size | 25 troops |
| Deployment positions | 60 perimeter tiles |
| Action space | `Discrete(246)` |
| Battle limit | 300 simulation ticks |
| Safety guard | 70 agent decisions |
| Render mode | `rgb_array` |

**One `step()` is one agent decision, not necessarily ten simulation ticks.** Purchasing and finishing shopping advance no combat time. Each deployment or wait advances up to ten ticks, stopping early if the attack ends.

An episode terminates when all scoring buildings are destroyed, no live or reserve troops remain, or the 300-tick battle limit is reached. The separate 70-decision safety guard produces truncation instead of termination.

## Action space

| Action ID | Meaning |
|---|---|
| 0 | Buy tank |
| 1 | Buy melee |
| 2 | Buy ranged |
| 3 | Buy breaker |
| 4 | Finish shopping |
| 5–244 | Deploy one troop at one perimeter tile |
| 245 | Wait |

Deployment actions use `5 + troop_index * 60 + tile_index`. Troop indices follow `tank`, `melee`, `ranged`, `breaker`. Perimeter indices run clockwise from the top-left corner, without duplicating corners.

The package exports `deployment_action(troop_type, tile_index)` and `describe_action(action)` to construct and interpret actions. An action produced by the helper is still subject to the current legal-action mask.

## Observation space

Observations are a dictionary with two arrays. Both use `float32` values bounded by `[0, 1]`.

| Key | Shape | Contents |
|---|---|---|
| `grid` | `(8, 16, 16)` | Spatial channels listed below |
| `vector` | `(9,)` | Army, budget, time and progress summaries |

### Grid channels

| Channel | Meaning |
|---|---|
| 0 | Town Hall locations |
| 1 | Cannon locations |
| 2 | Mortar locations |
| 3 | Storage locations |
| 4 | Wall locations |
| 5 | Building health / maximum health |
| 6 | Live defence coverage count / 5 |
| 7 | Troop density / 8, clipped at 1 |

### Summary vector

| Index | Meaning |
|---|---|
| 0–3 | Reserve tank, melee, ranged and breaker counts, each divided by 25 |
| 4 | Remaining resource budget / 200 |
| 5 | Remaining battle time / 300 |
| 6 | Phase: 0 for shopping, 1 for battle |
| 7 | Destroyed building-value fraction |
| 8 | Live troop count / 25 |

The observation is **partially observable**: it does not contain each troop's type, health, target or movement progress. Two simulator states can therefore produce the same observation while having different future outcomes.

## Rewards and evaluation information

The step reward is the sum of:

| Component | Value |
|---|---|
| Newly destroyed scoring value | Value destroyed on this step / initial scoring value |
| Half-destruction milestone | +0.25 once |
| Town Hall destroyed | +0.25 once |
| Full clear | +0.25 once |
| Army purchase cost | −0.30 × spent / 200, charged once when shopping finishes |

Walls contribute zero scoring value. Wall shaping is disabled. With `fixed_army=True`, the cost is charged on the first battle action.

**Destruction measures building value, not win rate or the number of buildings destroyed.** For example, destroying buildings worth 100 points out of an initial 155 points gives `100 / 155 ≈ 0.6452`, or 64.52% destruction.

Useful `info` fields include `destruction` (fraction from 0 to 1), `town_hall`, `full_clear`, `finish_reason`, `tick`, `decisions`, `spent`, `reward_components`, `reward_totals`, `episode_return`, `action_mask`, and `settings_hash`.

## Layouts and reproducibility

| Motif | Layout structure | Included by default? |
|---|---|---|
| `closed_ring` | Closed wall ring | Yes |
| `ns_gates` | Ring with north and south gates | Yes |
| `ew_gates` | Ring with east and west gates | Yes |
| `layered_core` | Inner ring with an outer L-shaped wall | No; select explicitly |

To use the layered layout:

```python
import gymnasium as gym
import gymnasium_base_siege

env = gym.make("BaseSiege-v0", motifs=("layered_core",), base_seed=42)
observation, info = env.reset(seed=0)
env.close()
```

- `reset(seed=...)` seeds environment sampling. It does not separately seed `action_space.sample()`.
- `env.action_space.seed(...)` seeds random action-space sampling.
- `base_seed` fixes the procedural layout seed. Also select a single motif to fix both inputs to layout generation.
- `pool` accepts a nonempty list of `BaseSpec` layouts.
- `reset(options={"spec": spec})` selects an explicit `BaseSpec`.
- `env.unwrapped.base_spec` gives the active base specification; Gymnasium's `env.spec` remains registration metadata.

`BaseSpec` and `generate_base` are exported by the package. Custom specifications must satisfy the game geometry; the public API is not a comprehensive validator for arbitrary layout files.

The original accidental-access guard remains: layout seeds **11000–11999** require explicit `allow_final=True`. Automatic training-layout seed sampling uses 0–9999. Generated layouts are not automatically filtered using the capstone's difficulty-selection procedure.

## Rendering and snapshots

After installing the `render` extra, save the following as `render_base.py` and run `python render_base.py` in its folder. It writes `base_siege.png` there without opening a window.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium_base_siege

env = gym.make(
    "BaseSiege-v0",
    render_mode="rgb_array",
    base_seed=42,
    motifs=("closed_ring",),
)
env.reset(seed=0)
frame = env.render()
plt.imsave("base_siege.png", frame)
env.close()
```

`env.render()` returns an RGB NumPy array when `render_mode="rgb_array"`. With no render mode, it returns `None`.

For a JSON-compatible state snapshot, use `env.unwrapped.snapshot()` after reset. Constructor option `record=True` keeps decision-level snapshots in `env.unwrapped.history`; `record_ticks=True` additionally records presentation frames in `visual_history`. Recording can increase memory use and is off by default. The separate capstone web interface is not installed by this package.

## Additional constructor options

| Option | Purpose |
|---|---|
| `fixed_army=True` | Skip purchasing and start with the fixed heuristic army: 3 tanks, 6 melee, 4 ranged, 3 breakers |
| `record=True` | Retain decision-level snapshots |
| `record_ticks=True` | Retain additional frames during simulation |
| `render_mode="rgb_array"` | Enable image rendering |
| `allow_final=True` | Explicitly permit layout seeds in the protected range |

If custom troop costs make the fixed army exceed the 200-resource budget, construction is rejected. The default fixed army costs 199 resources.

## Known limitations

- Units may retain an inaccessible preferred target. **No reachable-target fallback has been added.** This is separate from the tank's fallback to other building types after all defences are destroyed.
- Movement uses four-neighbour A*, while attack ranges use Euclidean distance. Walls and buildings do not block shots.
- Observations omit some internal simulator state and do not explicitly describe all customized statistics.
- Board size, budget, time limit, target rules and reward coefficients are not public constructor settings in version 0.1.0.
- Changing troop or defence statistics changes the task. Scores from different settings should not be treated as the same benchmark.
- The related capstone web application's manual mode has different deployment timing and is not a controlled human-versus-agent benchmark.
- Python support is currently restricted to 3.12. The recorded package validation was performed on Linux; it is not a claim of testing every operating system.

### Gymnasium checker behaviour

Gymnasium's passive environment checks remain enabled. The standalone `check_env` samples unmasked actions, which can trigger Base Siege's intentional `ValueError` for illegal actions.

The included tests use a **test-only masked sampler** with `check_env` and check rendering separately. This is not a claim that an unmodified `check_env` run passes, and the environment does not silently replace illegal actions with waits.

## Development and tests

To work on the source, run the following in a terminal from the directory where you want to create the repository checkout:

```bash
git clone https://github.com/vaibhavbai1/gymnasium-base-siege.git
cd gymnasium-base-siege
```

Using a Python 3.12 environment, run the remaining commands **inside the cloned repository**, beside `pyproject.toml`:

```bash
python -m pip install -e ".[render,dev]"
python -m pytest -q
python examples/random_agent.py
python examples/custom_settings.py
```

The package preparation checks passed **182 tests** with Gymnasium 1.3.0 and 1.4.0 on Python 3.12.14/Linux. Default trajectories also matched the pinned original simulator across **24 complete attacks and 507 decisions**. These are software regression checks, not new agent-performance experiments or an exhaustive proof of equivalence.

The published PyPI 0.1.0 package was separately installed in a fresh environment and checked with complete default-setting and custom-setting attacks.

See [validation details](docs/VALIDATION.md), [source changes](docs/CHANGES.md), and [publishing instructions](docs/PUBLISHING.md). For a new release, build and check its distributions from the repository root:

```bash
python -m build
python -m twine check dist/*
```

## Project background and attribution

The public package is derived from the [Base Siege UTS capstone environment](https://github.com/Reinforcement-Learning-UTS/base-siege-dihedrals). The original source revision and simulator-file hashes are recorded in [PROVENANCE.json](PROVENANCE.json).

The capstone investigated PPO, Double DQN and TRPO. Their training pipelines, checkpoints, frozen evaluation pools, logs and results are not bundled with this environment package. The public package adds reusable packaging, Gymnasium registration and configurable troop/defence statistics while retaining the original default game rules.

**Public package author and maintainer:** Vaibhav Bairathi.

## License

Released under the [MIT License](LICENSE).

Copyright (c) 2026 Vaibhav Bairathi.
