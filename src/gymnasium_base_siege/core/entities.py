"""
Base Siege - entity classes.

Two dataclasses: Building and Troop. Both hold mutable state (hp, position,
target) and read immutable stats from config by type.

Section references are to BUILD_ORDER_v2.md §6 Stage 2.

--------------------------------------------------------------------------
THE ONE PROBLEM THIS MODULE EXISTS TO SOLVE
--------------------------------------------------------------------------
Fractional speed without floating-point drift.

A tank moves 0.10 tiles per tick. The naive implementation stores position as
a float and adds 0.10 every tick. After 300 ticks the position is not 30.0,
and - worse - it is not *reproducibly* not-30.0 across machines. The whole
seed-split protocol depends on identical bases producing identical episodes,
so this is a correctness requirement, not a tidiness one.

BUILD_ORDER's prescribed fix is a movement-credit accumulator: keep position
as integers, accumulate speed as a float, step one tile when the accumulator
crosses 1.0, subtracting 1.0 rather than resetting.

That fixes position drift and leaves credit drift untouched. Demonstrably:

    >>> c = 0.0
    >>> for _ in range(10): c += 0.10
    >>> c >= 1.0
    False

0.10 added ten times is 0.9999999999999999. A tank with speed 0.10 would
never take its first step, and Stage 2 acceptance test 1 - which demands
exactly one tile after ten accumulations - fails against the very
implementation the plan prescribes.

So the accumulator here is INTEGER. Speed is converted once, at construction,
to an integer numerator over config.SPEED_SCALE. Accumulation is integer
addition. There is no float anywhere in the movement path, which means no
drift, no epsilon comparisons, and bit-identical behaviour on every machine.

Public-package extension: optional per-entity stat dictionaries are copied at spawn.
Without overrides, stats are read through properties from the original defaults. Copying
means a change to the config table silently fails to reach entities that were
already constructed. The cost is one dict lookup per access; if Checkpoint A
profiling shows it matters, cache in __post_init__ - but measure first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gymnasium_base_siege.core import config as C


# ===========================================================================
# Building
# ===========================================================================


@dataclass
class Building:
    """
    A static structure: town hall, cannon, mortar, storage, or wall.

    Position is immutable after construction - buildings never move - but it
    is not enforced at runtime, because a frozen dataclass would also freeze
    hp, and hp is the whole point.

    Buildings deliberately carry NO entity_id, unlike troops. Two buildings
    can never occupy the same tile, so (row, col) is already a unique and
    deterministic sort key for "nearest building" tie-breaks. Troops CAN share
    a tile - several can be deployed on one perimeter tile - so they need an
    explicit id. See O2 in the decision register; the tie-break rule itself is
    Stage 5's to close, this module only guarantees a key exists.
    """

    building_type: str
    position: tuple[int, int]
    hp: int
    max_hp: int
    target: int | None = None
    cooldown_remaining: int = 0   # ticks until this defence may fire again
    stats_override: dict | None = field(default=None, repr=False)

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def spawn(cls, building_type: str, position: tuple[int, int], *, stats=None) -> "Building":
        """
        Construct at full health from the config stat table.

        Preferred over calling the constructor directly: it is the only place
        that knows hp and max_hp both start from the same table entry, so they
        cannot be wired up inconsistently at a call site.
        """
        hp = (C.BUILDING_STATS[building_type] if stats is None else stats)["hp"]
        return cls(building_type=building_type, position=position, hp=hp, max_hp=hp,
                   stats_override=None if stats is None else dict(stats))

    # -- stats, read from config, never copied ------------------------------

    @property
    def _stats(self) -> dict:
        return C.BUILDING_STATS[self.building_type] if self.stats_override is None else self.stats_override

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def is_wall(self) -> bool:
        return self.building_type == C.WALL

    @property
    def value(self) -> int:
        """Score contribution. Zero for walls - structural, not a penalty."""
        return self._stats["value"]

    @property
    def damage_per_shot(self) -> int:
        """
        Damage from ONE shot, not per tick. The two differ for any defence
        with a cooldown above 1 - currently the mortar, at 45 every 5 ticks.
        Named for the shot rather than the tick so the mortar's stat cannot
        be misread as a continuous rate.
        """
        return self._stats["damage_per_shot"]

    @property
    def attack_cooldown(self) -> int:
        """Ticks between shots. 1 means fire every tick."""
        return self._stats["attack_cooldown"]

    @property
    def effective_dps(self) -> float:
        """Sustained damage per tick. What the balance tables should quote."""
        return self.damage_per_shot / self.attack_cooldown

    @property
    def attack_range(self) -> float:
        """May be fractional - the mortar sits at 4.5."""
        return self._stats["range"]

    @property
    def min_range(self) -> int:
        """Blind spot. Non-zero only for the mortar, and deliberately so."""
        return self._stats["min_range"]

    @property
    def splash_radius(self) -> float:
        return self._stats["splash_radius"]

    @property
    def is_defence(self) -> bool:
        return self.building_type in C.DEFENCE_TYPES

    @property
    def health_fraction(self) -> float:
        """
        Current HP as a fraction in [0.0, 1.0]. Feeds observation layer 5,
        which Gymnasium declares as Box(0, 1) and will reject at Stage 7 if
        this ever escapes the range. take_damage clamps so it cannot.
        """
        return self.hp / self.max_hp

    # -- mutation -----------------------------------------------------------

    def take_damage(self, amount: int) -> None:
        """
        Apply damage, clamping at zero.

        Negative HP has no meaning and would drive health_fraction below 0,
        out of the observation's declared Box(0, 1). Non-positive amounts are
        ignored rather than healing.
        """
        if amount <= 0:
            return
        self.hp = max(0, self.hp - amount)


# ===========================================================================
# Troop
# ===========================================================================


@dataclass
class Troop:
    """
    An attacker: tank, melee, ranged, or breaker.

    position is an integer (row, col) at all times. Fractional progress lives
    in movement_credit, which is an INTEGER in units of 1/SPEED_SCALE of a
    tile - see the module docstring for why that is not a stylistic choice.

    entity_id is not decoration. Every "nearest X" rule needs a deterministic
    tie-break, or the shuffle-invariance test at Stage 5 fails intermittently,
    which is the worst kind of failure to chase.

    REFINEMENT OF THE STAGE 2 CONTRACT. `target` holds a TILE, not an int.
    The contract typed it `int | None` on the assumption that everything
    carries an entity id, but buildings deliberately do not - two buildings
    can never share a tile, so position already identifies one uniquely.
    Each entity is therefore identified by whatever is actually unique about
    it: a troop by its id, a building by its tile. `Building.target` stays an
    int, because troops CAN stack and position does not identify them.
    """

    troop_type: str
    position: tuple[int, int]
    hp: int
    max_hp: int
    entity_id: int
    movement_credit: int = 0
    target: tuple[int, int] | None = None
    cached_path: list[tuple[int, int]] | None = field(default=None)
    stats_override: dict | None = field(default=None, repr=False)

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def spawn(
        cls, troop_type: str, position: tuple[int, int], entity_id: int, *, stats=None
    ) -> "Troop":
        """Construct at full health from the config stat table."""
        hp = (C.TROOP_STATS[troop_type] if stats is None else stats)["hp"]
        return cls(
            troop_type=troop_type,
            position=position,
            hp=hp,
            max_hp=hp,
            entity_id=entity_id,
            stats_override=None if stats is None else dict(stats),
        )

    # -- stats, read from config, never copied ------------------------------

    @property
    def _stats(self) -> dict:
        return C.TROOP_STATS[self.troop_type] if self.stats_override is None else self.stats_override

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def speed(self) -> float:
        """
        Tiles per tick, as written in config. Provided for reporting and for
        pathfinding cost arithmetic (open_tile_cost = 1 / speed).

        NOT used by the movement loop - see speed_units.
        """
        return self._stats["speed"]

    @property
    def speed_units(self) -> int:
        """
        Speed as an exact integer numerator over config.SPEED_SCALE.

        round() rather than int(): 0.16 * 100 is 16.000000000000004 in binary
        floating point, and int() would truncate that to 16 by luck rather
        than by design. A speed whose scaled value is not within half a unit
        of an integer is a config error, guarded by a test rather than a
        runtime assert so the tick loop stays free of checks.
        """
        return round(self._stats["speed"] * C.SPEED_SCALE)

    @property
    def damage_per_tick(self) -> int:
        return self._stats["damage_per_tick"]

    @property
    def attack_range(self) -> int:
        return self._stats["range"]

    @property
    def walls_only(self) -> bool:
        """True for the breaker: it physically cannot damage a non-wall."""
        return self._stats["walls_only"]

    @property
    def dies_after_attack(self) -> bool:
        """True for the breaker: one wall, then it is gone."""
        return self._stats["dies_after_attack"]

    @property
    def target_rule(self) -> str:
        return self._stats["target_rule"]

    @property
    def cost(self) -> int:
        return self._stats["cost"]

    @property
    def min_range(self) -> int:
        """
        Always zero. Only the mortar has a blind spot, but combat's in_range
        check reads min_range off whatever it is handed, so troops expose it
        too rather than making every call site special-case its argument.
        """
        return 0

    @property
    def health_fraction(self) -> float:
        return self.hp / self.max_hp

    @property
    def movement_fraction(self) -> float:
        """
        Progress toward the next tile, in [0.0, 1.0). For the renderer's
        interpolated movement only. Read-only, and never fed back into the
        accumulator - the moment a float re-enters the movement path, the
        drift this module exists to prevent comes back with it.
        """
        return self.movement_credit / C.SPEED_SCALE

    # -- mutation -----------------------------------------------------------

    def accumulate_movement(self) -> bool:
        """
        Advance the movement accumulator by one tick.

        Returns True when a one-tile step is now due. The caller performs the
        step: this class knows when to move, not where to, because "where"
        needs the path and the grid.

        Subtracts one full tile rather than resetting to zero. Resetting
        discards the remainder - a melee troop at 0.16 accumulates to 1.12 and
        would throw away 0.12 every tile, arriving roughly 10% late. That is
        invisible until you find yourself wondering why melee underperforms.

        One step at most per call. Every speed in config is below 1.0 tile per
        tick, so that is sufficient; the invariant is guarded by a test.
        """
        self.movement_credit += self.speed_units
        if self.movement_credit >= C.SPEED_SCALE:
            self.movement_credit -= C.SPEED_SCALE
            return True
        return False

    def take_damage(self, amount: int) -> None:
        """Apply damage, clamping at zero. See Building.take_damage."""
        if amount <= 0:
            return
        self.hp = max(0, self.hp - amount)
