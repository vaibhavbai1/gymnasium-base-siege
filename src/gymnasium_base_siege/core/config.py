"""
Base Siege - configuration constants.

Every tunable number in the project lives here. If a magic number appears
anywhere else in the codebase, that is a bug (BUILD_ORDER §2).

This module contains constants only: no functions, no classes, no I/O.
Some constants are *derived* from others by a module-level expression
(e.g. TOTAL_BASE_VALUE). That is deliberate. A derived constant cannot
drift out of sync with the table it came from; a hand-typed one can.

Section references are to BASE_SIEGE_PROJECT_PLAN_v3.md.
"""

# ---------------------------------------------------------------------------
# Units and conventions  (plan §6.2, §6.9)
# ---------------------------------------------------------------------------
#
# TICK       one frame of the simulator. Every stat below is per tick.
# TIMESTEP   one agent decision. One timestep = TICKS_PER_DECISION ticks.
#            This is what Stable-Baselines3 counts.
#
# Two different distance metrics are in use, on purpose:
#   MOVEMENT  four-neighbour A*, so travel distance is Manhattan-like.
#   COMBAT    Euclidean, so ranges and splash are circular.
# Mixing them is a bug. Both are named below so neither is implicit.

DISTANCE_METRIC_COMBAT = "euclidean"
PATHFINDING_NEIGHBOURS = 4

# ---------------------------------------------------------------------------
# Line of sight
# ---------------------------------------------------------------------------
#
# DECISION RECORDED 10 August 2026. Not specified in the v3 plan; must be
# written into §6.8 before Part B freeze.
#
# NOTHING BLOCKS A SHOT. Range alone decides what an attacker can hit.
#
# The asymmetry that matters is produced by the range table, not by a blocking
# rule: the ranged troop is the only troop with range above 1, so it is the
# only one that can attack a building across a wall. Tank, melee and breaker
# must reach an adjacent tile, which means breaking through or going around.
# Defences fire over their own walls too, so the approach is never free.
#
# The alternative - solid tiles block shots - was considered and rejected. It
# would seal every defence inside a closed ring, unable to fire outward at all,
# which makes the approach phase cost nothing and inverts motif difficulty.
#
# Kept as explicit constants rather than an absent feature, so the decision
# appears in the run manifest and cannot be silently reversed.

WALLS_BLOCK_LINE_OF_SIGHT = False
BUILDINGS_BLOCK_LINE_OF_SIGHT = False

# ---------------------------------------------------------------------------
# Map and episode  (plan §6.1)
# ---------------------------------------------------------------------------

GRID_SIZE = 16

RESOURCE_BUDGET = 200
MAX_ARMY_SIZE = 25            # exactly BUDGET / cheapest troop cost; see note below

BATTLE_TICK_LIMIT = 300       # terminates, does not truncate (§7.8)
TICKS_PER_DECISION = 10       # frame skip / action repeat (§6.2)
MAX_BATTLE_DECISIONS = BATTLE_TICK_LIMIT // TICKS_PER_DECISION   # 30
MAX_DECISIONS_PER_EPISODE = 70   # safety guard only; this one truncates (§7.8)

PHASE_SHOPPING = 0
PHASE_BATTLE = 1

# ---------------------------------------------------------------------------
# Deployment ring  (plan §6.1, §7.4)
# ---------------------------------------------------------------------------
#
# The deployment zone is the entire grid perimeter: 4 * 16 - 4 = 60 tiles.
#
# ORDERING IS LOAD-BEARING. Tiles run CLOCKWISE from the top-left corner.
# With that ordering a 90 degree rotation of the board is exactly a cyclic
# shift of +15 in tile index, which makes the Stage 12 symmetry mapping
# tractable. A reflection is *not* a cyclic shift - it is an index reversal
# about an axis - which is why Stage 12 requires an explicit bijection test
# rather than trusting arithmetic. Do not reorder this list.

_PERIMETER_TOP = [(0, c) for c in range(GRID_SIZE)]                            # 16
_PERIMETER_RIGHT = [(r, GRID_SIZE - 1) for r in range(1, GRID_SIZE)]           # 15
_PERIMETER_BOTTOM = [(GRID_SIZE - 1, c) for c in range(GRID_SIZE - 2, -1, -1)]  # 15
_PERIMETER_LEFT = [(r, 0) for r in range(GRID_SIZE - 2, 0, -1)]                # 14

DEPLOY_TILES = tuple(
    _PERIMETER_TOP + _PERIMETER_RIGHT + _PERIMETER_BOTTOM + _PERIMETER_LEFT
)
N_DEPLOY_TILES = len(DEPLOY_TILES)          # 60
ROTATION_INDEX_SHIFT = GRID_SIZE - 1        # 15; one 90-degree turn, clockwise

# ---------------------------------------------------------------------------
# Buildable region  (plan §6.1 "central region")
# ---------------------------------------------------------------------------
#
# The plan describes the buildable area only in words, so these bounds are a
# design decision made here rather than a value copied from the plan.
#
# BUILDABLE_MIN/MAX are the HARD bound: a building may never occupy a
# deployment tile, because a troop must be able to spawn on every perimeter
# tile. That is the invariant Stage 1 tests.
#
# The tighter, motif-dependent region - "inside this base's wall ring" - is
# derived in generator.py from the ring geometry below, because it depends on
# which motif is being built. Config holds geometry; the generator holds the
# rule that turns geometry into a placement region.

BUILDABLE_MIN = 1
BUILDABLE_MAX = GRID_SIZE - 2     # 14 inclusive

# ---------------------------------------------------------------------------
# Troops  (plan §6.3)
# ---------------------------------------------------------------------------
#
# All values per tick. Speed is tiles/tick, so 1/speed is ticks per tile -
# which is the quantity pathfinding costs must be expressed in (§6.9).
#
# On the stat relationships, both of which are deliberate:
#
#   ranged range (4) > cannon range (3)
#       Creates a standoff annulus where a ranged troop can shoot a cannon
#       and the cannon cannot reply. Without it the ranged troop is a strictly
#       worse melee troop (less HP, less damage, higher cost) and rational
#       shopping never buys one. Asserted in tests/test_config.py.
#
#   mortar min_range (2) > 0
#       Creates a blind spot the mortar cannot fire into, so danger is not
#       monotone in distance and placement is a spatial problem rather than a
#       scalar ordering. Asserted in tests/test_config.py.

TROOP_TYPES = ("tank", "melee", "ranged", "breaker")

TROOP_STATS = {
    "tank": {
        "hp": 900,
        "damage_per_tick": 8,
        "range": 1,
        "speed": 0.10,
        "cost": 25,
        "target_rule": "nearest_defence",
        "walls_only": False,
        "dies_after_attack": False,
    },
    "melee": {
        "hp": 120,
        "damage_per_tick": 15,
        "range": 1,
        "speed": 0.16,
        "cost": 8,
        "target_rule": "nearest_building",
        "walls_only": False,
        "dies_after_attack": False,
    },
    "ranged": {
        "hp": 90,
        "damage_per_tick": 12,
        "range": 4,
        "speed": 0.14,
        "cost": 10,
        "target_rule": "nearest_building",
        "walls_only": False,
        "dies_after_attack": False,
    },
    "breaker": {
        "hp": 60,
        "damage_per_tick": 350,      # walls only; 350 > wall HP 300, so one strike, one wall
        "range": 1,
        "speed": 0.22,
        "cost": 12,
        "target_rule": "nearest_wall",
        "walls_only": True,
        "dies_after_attack": True,
    },
}

# Movement credit resolution.
#
# DECISION RECORDED 24 August 2026. Speeds above are written as decimals for
# readability, but they must NEVER be accumulated as floats.
#
# BUILD_ORDER Stage 2 prescribes a float accumulator to avoid float *position*
# drift. It does not avoid float *credit* drift, and its own acceptance test 1
# fails because of it: 0.10 added ten times is 0.9999999999999999 in IEEE-754
# double, which never crosses a 1.0 threshold. The troop stands still forever.
#
# The fix is to leave floating point out of the loop entirely. Speed is
# converted once, at construction, into an integer numerator over SPEED_SCALE,
# and the accumulator is integer addition - exact, and identical on every
# machine, which is what the seed-split protocol actually requires.
#
# SPEED_SCALE = 100 because every speed above is given to two decimal places.
# Add a three-decimal speed and this must become 1000; the guard test in
# tests/test_entities.py fails loudly if you forget.
SPEED_SCALE = 100

# What an attacker does when its preferred target class is empty.
#
# DECISION RECORDED 15 August 2026, found by playing the prototype: with no
# fallback, a tank whose target rule is "nearest_defence" stands idle for the
# rest of the battle once all five defences are dead - absorbing nothing,
# contributing nothing, while 900 HP of army sits unused.
#
# A tank falls back to the nearest non-wall building. A breaker does NOT:
# walls_only is True, so it physically cannot damage a building, and idling
# is the honest cost of overbuying breakers - information the agent should
# learn rather than be insulated from.
#
# None means "no fallback; wait". Combat must implement this explicitly, and
# must not silently treat an empty target pool as a crash or a no-op.
TARGET_RULE_FALLBACK = {
    "nearest_defence": "nearest_building",
    "nearest_building": None,
    "nearest_wall": None,
}

TROOP_COSTS = {t: TROOP_STATS[t]["cost"] for t in TROOP_TYPES}
CHEAPEST_TROOP_COST = min(TROOP_COSTS.values())      # 8

# Troops that can attack a building on the far side of a wall. Derived from
# range, not declared: any troop with range 1 must stand adjacent to its
# target, so a wall between them is a wall it has to get through.
WALL_CROSSING_TROOPS = tuple(t for t in TROOP_TYPES if TROOP_STATS[t]["range"] > 1)

# ---------------------------------------------------------------------------
# Buildings and scoring  (plan §6.4)
# ---------------------------------------------------------------------------
#
# WALLS SCORE ZERO. This is structural, not a reward penalty. 36 walls at
# 1 point each would collectively outvalue the Town Hall while being the
# easiest thing on the map to destroy, so wall farming would be the optimal
# policy rather than a bug to be discouraged (§6.4).

BUILDING_TYPES = ("town_hall", "cannon", "mortar", "storage", "wall")
WALL = "wall"
DEFENCE_TYPES = ("cannon", "mortar")

# DEFENCE FIRING MODEL, REVISED 24 August 2026.
#
# Defences now have a FIRE RATE. `damage_per_shot` is what one shot does;
# `attack_cooldown` is how many ticks pass between shots. A cooldown of 1 is
# the old behaviour - fire every tick - and the cannon keeps it.
#
# The mortar does not. It was firing 9 damage every single tick, which made it
# a continuous beam rather than an artillery piece: there was no moment of
# safety after a shell landed, so timing an approach was pointless. It now
# fires once every 5 ticks for 45, and reaches 5 tiles rather than 6.
#
# 45 / 5 = 9, so its damage per tick is UNCHANGED. That is deliberate. Slowing
# the rate while holding total output isolates the thing being changed - the
# burst pattern - from the thing that is not, which is how much damage the
# defence contributes over an approach. Two shells now kill a ranged troop or
# a breaker and three kill a melee troop, so there is a real gap between
# shells that a player, or an agent, can move through.
#
# To nerf the mortar outright rather than only reshape it, lower
# damage_per_shot; nothing else needs to move.

BUILDING_STATS = {
    "town_hall": {
        "hp": 1500, "damage_per_shot": 0, "range": 0, "attack_cooldown": 1,
        "min_range": 0, "splash_radius": 0.0, "value": 30,
    },
    "cannon": {
        "hp": 600, "damage_per_shot": 5, "range": 3, "attack_cooldown": 1,
        "min_range": 0, "splash_radius": 0.0, "value": 15,
    },
    "mortar": {
        "hp": 500, "damage_per_shot": 45, "range": 5.0, "attack_cooldown": 5,
        "min_range": 2, "splash_radius": 1.5, "value": 20,
    },
    "storage": {
        "hp": 800, "damage_per_shot": 0, "range": 0, "attack_cooldown": 1,
        "min_range": 0, "splash_radius": 0.0, "value": 10,
    },
    "wall": {
        "hp": 300, "damage_per_shot": 0, "range": 0, "attack_cooldown": 1,
        "min_range": 0, "splash_radius": 0.0, "value": 0,
    },
}

# Splash falloff.
#
# A mortar shell covers the 3x3 block around the tile it lands on. The tile it
# actually lands on takes the full shot; the eight surrounding tiles take this
# fraction of it.
#
# Uniform splash made the mortar's aim irrelevant - landing a shell one tile
# off cost nothing - so there was no difference between a well-placed shell
# and a lucky one. With falloff, a clustered army is punished harder than a
# spread one at the point of impact, which is a spatial property the agent can
# actually learn to avoid.
#
# Rounded, and floored at 1: a splash hit must never silently become free.
SPLASH_FALLOFF = 0.4

# Fixed per base. Walls are not listed here: their count is set by the motif
# template, and they contribute nothing to the score in any case.
BUILDING_COUNTS = {
    "town_hall": 1,
    "cannon": 3,
    "mortar": 2,
    "storage": 4,
}

N_SCORING_BUILDINGS = sum(BUILDING_COUNTS.values())              # 10
N_DEFENCES = sum(BUILDING_COUNTS[t] for t in DEFENCE_TYPES)      # 5

# Derived, not typed. Changing a count or a value updates this automatically.
TOTAL_BASE_VALUE = sum(
    BUILDING_STATS[btype]["value"] * count
    for btype, count in BUILDING_COUNTS.items()
)                                                                 # 155

# ---------------------------------------------------------------------------
# Wall motifs  (plan §6.6)
# ---------------------------------------------------------------------------
#
# Motifs are templates drawn by a loop, never random scatter. A ring drawn by
# a loop is always a coherent ring; scattered walls produce incoherent bases
# and an incoherent difficulty distribution.
#
# Ring boxes are (row_min, col_min, row_max, col_max), inclusive. The ring is
# the boundary of that box, so a 10x10 box gives 4*10 - 4 = 36 wall tiles -
# matching the "~36" in the plan exactly.

MOTIFS = ("closed_ring", "ns_gates", "ew_gates", "layered_core")
TRAIN_MOTIFS = ("closed_ring", "ns_gates", "ew_gates")
DEV_MOTIFS = TRAIN_MOTIFS
TRANSFER_MOTIF = "layered_core"

CENTRED_RING_BOX = (3, 3, 12, 12)     # rows 3..12, cols 3..12; 36 wall tiles
GATE_WIDTH = 2                        # tiles removed per gate, centred on its edge

# Layered core - the structural transfer motif, EXCLUDED FROM TRAINING (§10.5).
#
# Two nested regions instead of one: an L-shaped outer wall covering only the
# north and west edges of the same 10x10 footprint (19 tiles), plus a 5x5 ring
# enclosing the Town Hall (16 tiles). 35 walls total.
#
# Replaces the offset ring from an earlier draft. A shifted ring is the same
# topology moved, so a policy trained on the closed ring transfers to it almost
# for free - a weak structural test. This is a genuinely different topology:
# one breaker no longer opens the base, and the unwalled south-east corner is a
# free route that leads past whatever defends that side. All eight of its
# orientations are distinct, which is what hypothesis 3 needs.
#
# It also restores meaning to the two §6.7 structural checks ("Town Hall inside
# the innermost enclosed region", "every enclosed compartment contains at least
# one building"), which are vacuous when every motif has a single region.

LAYERED_OUTER_BOX = (3, 3, 12, 12)
LAYERED_OUTER_SIDES = ("north", "west")   # adjacent, so one corner stays open
LAYERED_CORE_BOX = (6, 6, 10, 10)         # rows 6..10, cols 6..10; 16 wall tiles

# What lives inside the inner ring. CLOSES OPEN QUESTION O3.
#
# The Town Hall alone makes the core a passive shell: breach it and nothing
# fights back. A defence gives the compartment teeth, and a storage raises the
# value locked behind the second wall to 30 + 20 + 10 = 60 of 155. An attacker
# who strips only the outer region therefore caps at 95/155 = 61.3%, below the
# 85% difficulty ceiling - so scoring well REQUIRES breaching both walls, which
# is the double-breach problem this motif was chosen to pose.
#
# A MORTAR, NOT A CANNON. REVISED 24 August 2026, replacing the earlier D14.
#
# The old rationale was that a mortar "cannot defend the compartment it sits
# in": the core interior is 3x3, so a mortar at the centre reaches only 1.41,
# inside its own minimum range of 2. That is true and it is not the whole
# picture. Measured from the core centre at range 5:
#
#                     core interior   core wall ring   outer region
#     cannon (rng 3)       9 / 9          16 / 16          4 / 56
#     mortar (rng 5)       0 / 9          16 / 16         52 / 56
#
# The mortar covers every wall-ring tile - which is where every breach must
# happen, because a troop has to stand adjacent to a wall to break it - and it
# shells 52 of the 56 outer tiles as well. It is blind only to a troop that is
# already inside, after the breach has succeeded.
#
# That turns the core from a prize into an artillery emplacement: the attacker
# is under sustained fire for the ENTIRE outer-region fight, and the only way
# to stop it is to breach the inner ring. A cannon cannot create that pressure
# - it reaches 4 outer tiles. The layered core is the structural transfer
# motif, so a tactical problem the training motifs cannot pose is exactly what
# it should be contributing.
#
# The cost is real and should be stated rather than hidden: once a troop is
# inside the 3x3, nothing there fights back. That is a coherent risk/reward
# story - expensive to reach, defenceless once reached - not an oversight.
LAYERED_CORE_CONTENTS = ("town_hall", "mortar", "storage")

# ---------------------------------------------------------------------------
# Base difficulty validation  (plan §6.7)
# ---------------------------------------------------------------------------

DIFFICULTY_MIN = 0.25     # below this: reward too sparse to learn from
DIFFICULTY_MAX = 0.85     # above this: no headroom to show improvement
HEURISTIC_VALIDATION_SIDES = 4   # run the heuristic from each side, take the best

# ---------------------------------------------------------------------------
# Action space  (plan §7.4)
# ---------------------------------------------------------------------------
#
#   0 .. 3     buy tank / melee / ranged / breaker
#   4          finish shopping
#   5 .. 244   deploy: 5 + (troop_index * 60) + tile_index
#   245        wait

ACTION_BUY_FIRST = 0
ACTION_BUY_LAST = ACTION_BUY_FIRST + len(TROOP_TYPES) - 1        # 3
ACTION_FINISH_SHOPPING = ACTION_BUY_LAST + 1                     # 4
ACTION_DEPLOY_FIRST = ACTION_FINISH_SHOPPING + 1                 # 5
ACTION_DEPLOY_LAST = ACTION_DEPLOY_FIRST + len(TROOP_TYPES) * N_DEPLOY_TILES - 1   # 244
ACTION_WAIT = ACTION_DEPLOY_LAST + 1                             # 245
N_ACTIONS = ACTION_WAIT + 1                                      # 246

# ---------------------------------------------------------------------------
# Observation  (plan §7.2)
# ---------------------------------------------------------------------------
#
# Both observation components are declared Box(0, 1), so every value must be
# normalised. The normalisers live here so no module invents its own.
#
# The observation is PARTIAL, not Markov (§7.3). Layer 7 gives troop DENSITY
# only - no per-troop health, type, or target. Two states that look identical
# to the policy can have completely different futures.

N_GRID_LAYERS = 8

GRID_LAYER_TOWN_HALL = 0
GRID_LAYER_CANNON = 1
GRID_LAYER_MORTAR = 2
GRID_LAYER_STORAGE = 3
GRID_LAYER_WALL = 4
GRID_LAYER_BUILDING_HEALTH = 5
GRID_LAYER_DEFENCE_COVERAGE = 6
GRID_LAYER_TROOP_DENSITY = 7

VECTOR_DIM = 9
# index 0..3 remaining count per troop type, 4 remaining budget,
# 5 remaining battle time, 6 phase, 7 destroyed-value fraction, 8 live troops

MAX_DEFENCE_COVERAGE = N_DEFENCES         # normaliser for layer 6; 5
MAX_TROOP_DENSITY_PER_TILE = 8            # normaliser for layer 7; clipped above this

# ---------------------------------------------------------------------------
# Reward  (plan §7.6)
# ---------------------------------------------------------------------------
#
# The destruction term divides by the LIVE non-wall building value computed at
# reset, not by the literal 155. The two agree for a valid base, but computing
# it means a base with a different composition cannot silently mis-scale the
# reward. Never hardcode 155 as a divisor (BUILD_ORDER Stage 7 pitfall).

MILESTONE_HALF_DESTRUCTION = 0.25
MILESTONE_TOWN_HALL = 0.25
MILESTONE_FULL_CLEAR = 0.25
HALF_DESTRUCTION_THRESHOLD = 0.50
COST_PENALTY_WEIGHT = 0.30

# Contingency from §6.4: if the agent refuses to buy breakers because they earn
# no direct reward, raise this to about 0.005. It is EXCLUDED from the reported
# destruction score and must be written up if used. Default off.
WALL_SHAPING_BONUS = 0.0

REWARD_COMPONENT_KEYS = ("destruction", "milestone", "cost_penalty", "wall_shaping")

# ---------------------------------------------------------------------------
# Baseline agents  (plan §9.2, BUILD_ORDER Stage 8)
# ---------------------------------------------------------------------------

HEURISTIC_ARMY = {"tank": 3, "melee": 6, "ranged": 4, "breaker": 3}
HEURISTIC_ARMY_COST = sum(
    TROOP_COSTS[t] * n for t, n in HEURISTIC_ARMY.items()
)                                                                 # 199
HEURISTIC_DEPLOY_ORDER = ("breaker", "tank", "ranged", "melee")
RANDOM_AGENT_ACTION_SEEDS = tuple(range(10))   # §9.1: average 10 seeds per base

# ---------------------------------------------------------------------------
# Seed splits  (plan §10.4)
# ---------------------------------------------------------------------------
#
# Ranges are inclusive. FINAL_SEEDS must not be read by any development tool.
# The guard belongs in generator.py, evaluate.py and ui/app.py, not here.

TRAIN_SEED_RANGE = (0, 9_999)
DEV_SEED_RANGE = (10_200, 10_399)
FINAL_SEED_RANGE = (11_000, 11_999)
FINAL_STUDY_SEED_RANGE = (11_000, 11_199)     # used ONCE, after freeze

POOL_TRAIN_PATH = "bases/pool_train.json"
POOL_DEV_PATH = "bases/pool_dev.json"
POOL_FINAL_PATH = "bases/pool_final.json"

# ---------------------------------------------------------------------------
# Learning system  (plan §8.1, §8.2)
# ---------------------------------------------------------------------------
#
# GAMMA = 1.0, not 0.99. SB3 discounts per TIMESTEP, and shopping decisions do
# not advance game time. At 0.99, buying 20 troops discounts all subsequent
# battle reward by 0.99^20 ~= 0.82 while buying 8 discounts by 0.99^8 ~= 0.92,
# penalising a larger army purely for taking more decisions to assemble. The
# episode is finite and bounded, so 1.0 is well defined (§7.7).

PPO_GAMMA = 1.0
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 10
PPO_GAE_LAMBDA = 0.95
PPO_ENT_COEF = 0.01

CURRICULUM_PLACEMENT_TIMESTEPS = 32_768    # four complete vector rollouts
CURRICULUM_JOINT_TIMESTEPS = 229_376       # twenty-eight complete vector rollouts
TOTAL_TRAINING_TIMESTEPS = (
    CURRICULUM_PLACEMENT_TIMESTEPS + CURRICULUM_JOINT_TIMESTEPS
)                                                                 # 262,144

STUDY_LEARNER_SEEDS = tuple(range(10))
PILOT_PLACEMENT_TIMESTEPS = 32_768
PILOT_JOINT_TIMESTEPS = 24_576              # 57,344 total; development only
PILOT_LEARNER_SEED = 100                   # separate from the main learner seeds
DEV_BASES_PER_MOTIF = 4
DEV_DIAGNOSTIC_VIEWS = (0, 1, 4, 5)

DEV_EVAL_INTERVAL_TIMESTEPS = 25_000
N_TRAINING_ENVS = 4                          # SubprocVecEnv workers

FEATURE_CNN_CHANNELS = (32, 64)
FEATURE_FUSION_DIM = 256
POLICY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128

# ---------------------------------------------------------------------------
# Performance gates  (BUILD_ORDER Stage 6)
# ---------------------------------------------------------------------------

CHECKPOINT_A_TARGET_MS = 50
CHECKPOINT_A_LIMIT_MS = 200
