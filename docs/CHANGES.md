# Differences from the capstone source

Upstream: `Reinforcement-Learning-UTS/base-siege-dihedrals`, commit
`99cc45524aa33d0c26bdc4e48c8cc2ac1431d182`.

The public package is a separate extraction. The upstream checkout and frozen
results were not edited. No training or long evaluation was run.

## Packaging

- Namespaced the ten simulator modules under `gymnasium_base_siege.core` so they
  do not conflict with another project's generic `env` package.
- Added registration as `BaseSiege-v0`, optional RGB-render dependencies, examples,
  package metadata and release instructions.
- Omitted models, training algorithms, research pools, UI and logs.
- Preserved original source comments for provenance. Historical plan references
  inside copied modules refer to the capstone, not bundled public-package documents.

## Public API

- `BaseSiegeEnv` adds an int8 `action_mask` and `settings_hash` to reset/step info.
- RGB rendering respects `render_mode`; snapshots remain JSON serializable.
- The layout is stored as `base_spec`, preserving Gymnasium's reserved `spec`
  attribute for its registration metadata after reset.
- Added validated, copied `troop_stats` and `defence_stats` constructor overrides.
- Each instance owns its stat tables. No temporary global patching is used.
- `settings` returns a copy that can be saved as JSON and supplied to a new environment.

## Simulator adaptations for custom settings

- Entities can receive stat overrides at creation. HP and all combat properties
  read the instance-specific values when provided.
- The grid stores troop stats so both A* movement costs and wall-breaking costs
  use the customized speed/damage. Its heuristic uses the same speed as its costs.
- Battle creation supplies building stats; deployment supplies troop stats.
- A customized zero-damage defence also deals zero splash damage; the original
  minimum-one splash rounding remains in effect for positive damage settings.
- Purchasing, legal masks and fixed-army costs use the same custom cost table.
- Defence coverage in observations already reads live entity properties; custom
  attack ranges therefore propagate into observations.
- Snapshots carry custom defence ranges, which the renderer uses for range rings.

## Rules deliberately retained

The board, deployment indexing, observation shapes, budget, army cap, rewards,
target rules, seven-step combat order, terminal conditions and seed guard retain
their original defaults. No reachable-target fallback was introduced. Inaccessible
preferred targets can still cause units to stop making progress.

Customization does not expose map dimensions, scoring weights, target logic,
wall/other-building HP, budget, time limit or reward coefficients in this release.
Defences are stationary: their configurable firing interval is `attack_cooldown`,
not a movement speed. Custom stat sets describe new tasks and must not be presented
as the frozen capstone benchmark.
