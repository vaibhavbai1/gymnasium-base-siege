"""
Base Siege - the combat simulator.

Section references are to BUILD_ORDER_v2.md §6 Stage 5 and plan §6.8.

--------------------------------------------------------------------------
THE SEVEN-STEP TICK ORDER
--------------------------------------------------------------------------
Every tick does exactly this, in this order, without exception:

    1. remove dead troops and buildings
    2. invalidate cached paths if the board's shape changed
    3. retarget troops whose target is gone
    4. troops move, or add attacks to a pending-damage buffer
    5. defences add attacks to the same buffer
    6. apply all pending damage together
    7. check whether the episode has ended

STEP 6 IS THE ONE THAT MATTERS. Applying damage as you walk the unit list
means whichever unit happens to sit earliest in a Python list attacks before
it can be killed. Results then depend on list ordering - which is meaningless
as a game rule, and worse, makes episodes irreproducible across Python
versions where dict and set iteration differ. Buffering makes the tick
order-independent: every attack is computed against the same board state, and
two units that would kill each other both die.

Be ready to recite this. It is question 10 on the v3 viva checklist.

--------------------------------------------------------------------------
TARGETING
--------------------------------------------------------------------------
    Tank     nearest DEFENCE      walks past storages entirely
    Melee    nearest BUILDING     any non-wall
    Ranged   nearest BUILDING     range 4 - the only troop that hits across a wall
    Breaker  nearest WALL         dies after one strike; 350 > 300
    Cannon   nearest TROOP        range 3, fires every tick
    Mortar   nearest TROOP        range 2..4.5, one shell every 5 ticks,
                                  3x3 splash with falloff outside the impact tile

Distance is EUCLIDEAN (D5). Nothing blocks a shot (D7) - range alone decides
what can be hit, which is why the ranged troop's range of 4 is the entire
source of the wall-crossing asymmetry.

CLOSING O1 AND O2. Ties break deterministically, but not by the same key for
both directions, because the two entity kinds are not identified the same way:

    defence -> troop     ascending entity_id      troops can share a tile
    troop -> building    ascending (row, col)     buildings cannot

Any weaker tie-break - list order, a bare min() over unsorted input -
reintroduces exactly the ordering dependence step 6 exists to remove, and the
shuffle test then fails one run in twenty rather than every time, which is far
harder to diagnose than a clean failure.

--------------------------------------------------------------------------
THE MORTAR BLIND SPOT
--------------------------------------------------------------------------
min_range 2 leaves eight tiles a mortar cannot touch: its four orthogonal and
four diagonal neighbours. Four of those sit at distance 1, which is melee and
tank range, so a troop that reaches the blind spot kills the mortar while it
is helpless. Only the tank survives the crossing - that is what its 900 HP is
for. It is a delivery mechanism, not a damage source.

--------------------------------------------------------------------------
NECESSARY ADDITIONS TO THE STAGE 5 CONTRACT
--------------------------------------------------------------------------
`BattleState` gains `next_entity_id`, `spent`, `troops_total`, `last_attacks`,
`phase` and `total_base_value`. None is optional: the first is needed to issue
unique troop ids, and the rest are all fields the render.py snapshot schema
requires and cannot derive from the board alone.

`buildings` is a property over `grid.buildings` rather than a second list.
Storing the same buildings twice invites the two copies to disagree, which is
the failure mode config.py's single-source rule exists to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core.entities import Building, Troop
from gymnasium_base_siege.core.generator import BaseSpec
from gymnasium_base_siege.core.pathfinding import GridState, PathCache

Tile = tuple[int, int]

_NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ===========================================================================
# Geometry
# ===========================================================================


def euclidean(a: Tile, b: Tile) -> float:
    """Combat distance. Euclidean so ranges and splash are circular (D5)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def in_range(attacker, target) -> bool:
    """
    Can `attacker` hit `target` right now?

    Inclusive at BOTH ends. A mortar with min_range 2 fires at exactly 2 and
    not at 1.41; a cannon with range 3 fires at exactly 3 and not at 3.16.
    Both boundaries are pinned by tests because an off-by-one at either end
    silently reshapes every base's difficulty.
    """
    distance = euclidean(attacker.position, target.position)
    return attacker.min_range <= distance <= attacker.attack_range


def splash_targets(centre: Tile, radius: float, troops: list[Troop]) -> list[Troop]:
    """
    Every live troop within `radius` of `centre`, Euclidean.

    Radius 1.5 covers the full 3x3 block, because a diagonal neighbour sits at
    1.41. Using Manhattan distance here would reach only the four orthogonal
    neighbours and halve the mortar's effectiveness - a bug that looks like a
    balance problem rather than a metric problem.
    """
    return [
        troop
        for troop in troops
        if troop.is_alive and euclidean(centre, troop.position) <= radius
    ]


# ===========================================================================
# Damage buffer
# ===========================================================================


class DamageBuffer:
    """
    Damage accumulated during one tick and applied all at once in step 6.

    Keyed on id() because Building and Troop are unfrozen dataclasses, so
    Python sets __hash__ to None and they cannot be dict keys directly. The
    entity is carried alongside its total so nothing has to be looked up again.
    """

    def __init__(self) -> None:
        self._pending: dict[int, list] = {}

    def add(self, entity, amount: int) -> None:
        if amount <= 0:
            return
        slot = self._pending.get(id(entity))
        if slot is None:
            self._pending[id(entity)] = [entity, amount]
        else:
            slot[1] += amount

    def apply(self) -> None:
        for entity, amount in self._pending.values():
            entity.take_damage(amount)
        self._pending.clear()

    def __len__(self) -> int:
        return len(self._pending)


# ===========================================================================
# BattleState
# ===========================================================================


@dataclass
class BattleState:
    """One battle in progress. Mutated in place by step_tick."""

    grid: GridState
    troops: list[Troop] = field(default_factory=list)
    reserve: dict[str, int] = field(default_factory=dict)
    tick: int = 0
    path_cache: PathCache = field(default_factory=PathCache)
    destroyed_value: int = 0
    total_base_value: int = C.TOTAL_BASE_VALUE
    milestones: dict[str, bool] = field(
        default_factory=lambda: {"half": False, "town_hall": False, "full_clear": False}
    )
    finish_reason: str | None = None

    # -- additions the snapshot schema requires -----------------------------
    next_entity_id: int = 1
    spent: int = 0
    troops_total: int = 0
    phase: str = "battle"
    last_attacks: list[dict] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: BaseSpec, reserve: dict[str, int] | None = None,
                  *, troop_stats=None, building_stats=None):
        """
        Start a battle on a fresh board.

        `total_base_value` is computed from the LIVE non-wall buildings, never
        hardcoded to 155. The two agree for a valid base, but computing it
        means a base with a different composition cannot silently mis-scale
        the destruction fraction.
        """
        grid = GridState.from_spec(spec, troop_stats=troop_stats, building_stats=building_stats)
        total = sum(b.value for b in grid.buildings.values() if not b.is_wall)
        state = cls(grid=grid, total_base_value=total)
        if reserve:
            state.reserve = dict(reserve)
            stats = C.TROOP_STATS if troop_stats is None else troop_stats
            state.spent = sum(stats[t]["cost"] * n for t, n in reserve.items())
        return state

    # -- views --------------------------------------------------------------

    @property
    def buildings(self) -> list[Building]:
        """Contract-named view over the grid. Not a second copy."""
        return list(self.grid.buildings.values())

    @property
    def wall_state_version(self) -> int:
        """Contract name for GridState.topology_version. See D19."""
        return self.grid.topology_version

    @property
    def destroyed_fraction(self) -> float:
        if self.total_base_value <= 0:
            return 0.0
        return self.destroyed_value / self.total_base_value

    @property
    def troops_alive(self) -> int:
        return sum(1 for t in self.troops if t.is_alive)

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None


# ===========================================================================
# Deployment
# ===========================================================================


def deploy_troop(state: BattleState, troop_type: str, tile: Tile) -> None:
    """
    Put one troop from the reserve onto the board.

    Ids are issued in deployment order and never reused, so they are a stable
    total order for tie-breaking even after earlier troops have died.
    """
    if state.reserve.get(troop_type, 0) <= 0:
        raise ValueError(f"no {troop_type} in reserve")
    if tile not in C.DEPLOY_TILES:
        raise ValueError(f"{tile} is not a deployment tile")

    state.reserve[troop_type] -= 1
    state.troops.append(Troop.spawn(troop_type, tile, state.next_entity_id,
        stats=None if state.grid.troop_stats is None else state.grid.troop_stats[troop_type]))
    state.next_entity_id += 1
    state.troops_total += 1


# ===========================================================================
# Targeting
# ===========================================================================


def _candidates(state: BattleState, rule: str) -> list[Building]:
    live = [b for b in state.grid.buildings.values() if b.is_alive]
    if rule == "nearest_defence":
        return [b for b in live if b.is_defence]
    if rule == "nearest_building":
        return [b for b in live if not b.is_wall]
    if rule == "nearest_wall":
        return [b for b in live if b.is_wall]
    raise ValueError(f"unknown target rule: {rule!r}")


def acquire_target(state: BattleState, troop: Troop) -> Tile | None:
    """
    The tile of the building this troop should attack, or None to wait.

    Falls back per config.TARGET_RULE_FALLBACK when the preferred class is
    empty (D15): a tank whose defences are all dead switches to the nearest
    other building rather than standing idle with 900 HP of unused army. A
    breaker does NOT fall back - it is walls_only, so it physically cannot
    damage a building, and idling is the honest cost of overbuying breakers.
    """
    rule = troop.target_rule
    while rule is not None:
        options = _candidates(state, rule)
        if options:
            # Ties by ascending (row, col): a building's tile is unique, so it
            # is already a total order. No entity id needed.
            return min(
                options,
                key=lambda b: (euclidean(troop.position, b.position), b.position),
            ).position
        rule = C.TARGET_RULE_FALLBACK.get(rule)
    return None


def _acquire_defence_target(state: BattleState, defence: Building) -> Troop | None:
    """
    Nearest live troop inside the defence's range band, ties by ascending
    entity_id. Closes O1.

    Troops can occupy the same tile, so position is not a total order for
    them - the id is what makes this deterministic.
    """
    reachable = [
        t for t in state.troops if t.is_alive and in_range(defence, t)
    ]
    if not reachable:
        return None
    return min(
        reachable,
        key=lambda t: (euclidean(defence.position, t.position), t.entity_id),
    )


def _approach_goals(state: BattleState, target_tile: Tile) -> set[Tile]:
    """
    Where a troop must stand to attack `target_tile` at range 1.

    Longer-ranged troops stop early on their own: `in_range` is checked before
    movement every tick, so a ranged troop walking toward a building halts the
    moment it comes within 4. Pathing to the adjacent tiles rather than to
    every tile within range keeps the goal set at four instead of fifty.
    """
    row, col = target_tile
    goals = set()
    for d_row, d_col in _NEIGHBOUR_OFFSETS:
        tile = (row + d_row, col + d_col)
        if state.grid.in_bounds(tile) and not state.grid.is_blocked(tile):
            goals.add(tile)
    return goals


# ===========================================================================
# The tick
# ===========================================================================


def step_tick(state: BattleState) -> None:
    """
    Advance the battle by one tick. Mutates in place, returns None.

    Returning a fresh state each tick would allocate 300 times per episode for
    no benefit - the caller never wants the old one.
    """
    if state.is_finished:
        return

    buffer = DamageBuffer()
    attacks: list[dict] = []

    _step_1_remove_dead(state)
    _step_2_invalidate_paths(state)
    _step_3_retarget(state)
    _step_4_troops_act(state, buffer, attacks)
    _step_5_defences_act(state, buffer, attacks)
    _step_6_apply_damage(state, buffer)
    # Settle kills before terminal detection; there may never be another tick.
    _step_1_remove_dead(state)
    _step_2_invalidate_paths(state)
    state.tick += 1
    state.last_attacks = attacks
    _step_7_check_end(state)


def _step_1_remove_dead(state: BattleState) -> None:
    """
    Sweep corpses. Building removal also bumps the grid's topology version,
    which is what step 2 keys off.

    Scoring happens here, exactly once per building, because this is the only
    place a building transitions from present to absent.
    """
    for building in state.grid.purge_destroyed():
        if not building.is_wall:
            state.destroyed_value += building.value
            if building.building_type == "town_hall":
                state.milestones["town_hall"] = True

    state.troops = [t for t in state.troops if t.is_alive]


def _step_2_invalidate_paths(state: BattleState) -> None:
    """
    Drop stale routes when the board's shape has changed.

    The PathCache keys on topology_version, so old entries are already
    unreachable - clearing them only reclaims memory. Per-troop cached paths
    do need explicit clearing, since nothing else would notice they now walk
    through a wall that is no longer there, or around one that is.
    """
    if getattr(state, "_seen_version", None) == state.wall_state_version:
        return
    state._seen_version = state.wall_state_version
    state.path_cache.invalidate()
    for troop in state.troops:
        troop.cached_path = None


def _step_3_retarget(state: BattleState) -> None:
    """Any troop whose target has died picks a new one."""
    for troop in state.troops:
        if troop.target is None:
            troop.target = acquire_target(state, troop)
            continue
        building = state.grid.building_at(troop.target)
        if building is None or not building.is_alive:
            troop.target = acquire_target(state, troop)
            troop.cached_path = None


def _step_4_troops_act(
    state: BattleState, buffer: DamageBuffer, attacks: list[dict]
) -> None:
    """
    Each troop either attacks its target or moves one tile toward it.

    Iterated in ascending entity_id so the ORDER of iteration is fixed. That
    is belt and braces - step 6 already makes the outcome independent of it -
    but it also fixes the order attacks appear in the snapshot, which keeps
    rendered frames reproducible.
    """
    for troop in sorted(state.troops, key=lambda t: t.entity_id):
        if not troop.is_alive or troop.target is None:
            continue

        target = state.grid.building_at(troop.target)
        if target is None or not target.is_alive:
            continue

        # walls_only troops cannot damage a building even standing on top of
        # it. Without this a breaker that fell back would chew a town hall.
        if troop.walls_only and not target.is_wall:
            continue

        if in_range(troop, target):
            buffer.add(target, troop.damage_per_tick)
            attacks.append(
                {
                    "from": [troop.position[0], troop.position[1]],
                    "to": [target.position[0], target.position[1]],
                    "kind": "troop",
                    "damage": troop.damage_per_tick,
                }
            )
            if troop.dies_after_attack:
                # Buffered like any other damage, so a breaker and its wall die
                # in the same step rather than in an order that depends on
                # where the kill happened to be written.
                buffer.add(troop, troop.hp)
            continue

        _move_toward(state, troop, buffer, attacks)


def _move_toward(state: BattleState, troop: Troop, buffer, attacks) -> None:
    """Attack an obstructing wall, or accumulate credit for an open-tile move."""
    goals = _approach_goals(state, troop.target)
    if not goals:
        return

    path = state.path_cache.find_path_to_any(
        troop.troop_type, troop.position, goals, state.grid
    )
    troop.cached_path = path
    if path is None or len(path) < 2:
        return                      # already adjacent, or nowhere to go: wait
    obstacle = state.grid.building_at(path[1])
    if obstacle is not None and obstacle.is_alive:
        if obstacle.is_wall:
            buffer.add(obstacle, troop.damage_per_tick)
            attacks.append({"from": list(troop.position), "to": list(obstacle.position),
                            "kind": "troop", "damage": troop.damage_per_tick})
            if troop.dies_after_attack:
                buffer.add(troop, troop.hp)
        return
    if troop.accumulate_movement():
        troop.position = path[1]


def _step_5_defences_act(
    state: BattleState, buffer: DamageBuffer, attacks: list[dict]
) -> None:
    """
    Every live cannon and mortar fires once.

    Iterated by position, which is a total order over buildings, so the
    snapshot's attack list is stable run to run.
    """
    defences = sorted(
        (b for b in state.grid.buildings.values() if b.is_alive and b.is_defence),
        key=lambda b: b.position,
    )

    for defence in defences:
        # The cooldown ticks down whether or not there is anything to shoot,
        # so a mortar that has been idle is ready the instant a troop walks
        # into its band rather than owing five ticks of warm-up.
        if defence.cooldown_remaining > 0:
            defence.cooldown_remaining -= 1
            continue

        target = _acquire_defence_target(state, defence)
        defence.target = target.entity_id if target else None
        if target is None:
            continue

        if defence.splash_radius > 0:
            # Splash includes the primary target - it is at distance 0 from
            # itself - so it takes its damage once, here, not twice.
            hit = splash_targets(target.position, defence.splash_radius, state.troops)
        else:
            hit = [target]

        for troop in hit:
            if troop.position == target.position:
                damage = defence.damage_per_shot          # the impact tile
            else:
                damage = 0 if defence.damage_per_shot == 0 else max(
                    1, round(defence.damage_per_shot * C.SPLASH_FALLOFF)
                )
            buffer.add(troop, damage)
            attacks.append({"from": list(defence.position), "to": list(troop.position),
                            "kind": "defence", "damage": damage,
                            "attacker_type": defence.building_type,
                            "impact": list(target.position),
                            "target_id": troop.entity_id,
                            "splash": troop.position != target.position})

        defence.cooldown_remaining = defence.attack_cooldown - 1



def _step_6_apply_damage(state: BattleState, buffer: DamageBuffer) -> None:
    """
    Apply every attack computed this tick, simultaneously.

    This is the step the whole ordering discipline exists for. Everything
    above only READ the board; nothing above changed a hit point.
    """
    buffer.apply()


def _step_7_check_end(state: BattleState) -> None:
    """
    Set finish_reason if the battle is over.

    The 300-tick deadline TERMINATES, it does not truncate: running out of
    time is a real ending of this task, not an artificial cut, so the value
    function must not bootstrap past it. Only the 70-decision safety guard at
    Stage 7 truncates.
    """
    if state.destroyed_fraction >= C.HALF_DESTRUCTION_THRESHOLD:
        state.milestones["half"] = True

    scoring_left = [
        b for b in state.grid.buildings.values() if b.is_alive and not b.is_wall
    ]
    if not scoring_left:
        state.milestones["full_clear"] = True
        state.finish_reason = "full_clear"
        return

    if state.tick >= C.BATTLE_TICK_LIMIT:
        state.finish_reason = "tick_limit"
        return

    if not state.troops and not any(state.reserve.values()):
        state.finish_reason = "army_eliminated"


# ===========================================================================
# Snapshot
# ===========================================================================


def record_snapshot(state: BattleState) -> dict:
    """
    One JSON-serialisable frame, matching the schema at the top of render.py.

    That schema is the contract, and it was written first on purpose: the GUI
    at Stage 14 replays recorded snapshots rather than re-simulating, so the
    demo cannot diverge from the evidence a marker is shown. Everything below
    is a plain int, str, float or list - a whole episode can be written to
    disk and replayed without the simulator.
    """
    return {
        "tick": state.tick,
        "phase": state.phase,
        "destroyed_fraction": state.destroyed_fraction,
        "spent": state.spent,
        "troops_alive": state.troops_alive,
        "troops_total": state.troops_total,
        "reserve": {t: state.reserve.get(t, 0) for t in C.TROOP_TYPES},
        "buildings": [
            {
                "type": b.building_type,
                "row": b.position[0],
                "col": b.position[1],
                "hp": b.hp,
                "max_hp": b.max_hp,
            }
            for b in sorted(state.grid.buildings.values(), key=lambda b: b.position)
            if b.is_alive
        ],
        "troops": [
            {
                "type": t.troop_type,
                "row": t.position[0],
                "col": t.position[1],
                "hp": t.hp,
                "max_hp": t.max_hp,
                "entity_id": t.entity_id,
                "path": [[r, c] for r, c in t.cached_path] if t.cached_path else None,
            }
            for t in sorted(state.troops, key=lambda t: t.entity_id)
            if t.is_alive
        ],
        "attacks": list(state.last_attacks),
    }


def run_battle(state: BattleState, max_ticks: int | None = None) -> BattleState:
    """
    Step until the battle ends. Convenience for tests and Checkpoint A -
    the real environment steps in TICKS_PER_DECISION chunks.
    """
    limit = max_ticks if max_ticks is not None else C.BATTLE_TICK_LIMIT
    while not state.is_finished and state.tick < limit:
        step_tick(state)
    return state
