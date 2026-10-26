"""
Base Siege - four-neighbour A* pathfinding.

Section references are to BUILD_ORDER_v2.md §6 Stage 4.

--------------------------------------------------------------------------
THE ONE CORRECTION THIS MODULE EXISTS FOR
--------------------------------------------------------------------------
Path costs are in TICKS. Both terms, always.

    open_tile_cost = 1 / speed                        ticks to cross one tile
    wall_tile_cost = open_tile_cost + hp / dps        + ticks to break it

The naive formulation costs an open tile at 1 and a wall at hp/dps. That
compares one *grid step* against several dozen *wall-breaking ticks* as
though they were the same quantity. Troops then detour far too eagerly and
the breaker - whose entire purpose is to walk through walls - quietly
becomes worthless.

Concretely: a melee troop at speed 0.16 takes 6.25 ticks per tile. A ten-tile
detour costs 62.5 ticks, not 10.

The consequence is a clean closed form for how far a troop will walk to avoid
one wall (D9):

    break-even detour = speed * wall_hp / damage_per_tick

        melee    3.20 tiles      ranged   3.50 tiles
        tank     3.75 tiles      breaker  0.19 tiles

The breaker figure is the design working. It will not detour even one tile;
it walks straight through. Note that speed appears in that expression. Under
the buggy unit model it cancels out entirely, which is what acceptance test 3
detects.

--------------------------------------------------------------------------
WALLS ARE EXPENSIVE, NOT IMPASSABLE
--------------------------------------------------------------------------
A* treats a wall as a costly tile rather than a blocked one, so there is no
such thing as a structurally unattackable base. A perfectly sealed ring is
always solvable, just expensive. That removes an entire class of generator
validation - nothing has to prove a base is reachable, because every base is.

Non-wall buildings DO block. You cannot walk through a cannon.

--------------------------------------------------------------------------
FOUR NEIGHBOURS, NOT EIGHT
--------------------------------------------------------------------------
Eight-neighbour movement makes a diagonal step cost sqrt(2), which interacts
badly with tick-unit costs, and it lets a troop slip diagonally between two
wall corners - a rule nobody wants to define or defend in a viva. Four
neighbours keeps movement a clean lattice and matches config.PATHFINDING_NEIGHBOURS.

Heuristic is Manhattan distance x open_tile_cost. Admissible - it never
overestimates, because no tile costs less than an open one and movement is
4-neighbour - which is what makes A* optimal rather than merely fast.

--------------------------------------------------------------------------
DEVIATION FROM THE STAGE 4 CONTRACT, AND WHY
--------------------------------------------------------------------------
1. `GridState` is DEFINED HERE. The contract names it in `find_path`'s
   signature and no stage in BUILD_ORDER ever creates it. Defining it in the
   module that consumes it means the interface is shaped by a real caller
   rather than guessed at - the same argument that put render.py ahead of
   combat.py. Stage 5 will own and mutate it.

2. The cache version bumps on ANY building death, not only wall death. The
   contract says "an integer bumped every time a wall dies", which misses the
   case where a cannon is destroyed: its tile becomes passable, potentially
   opening a shortcut, and every cached path still routes around it. Bumping
   on any death costs at most ten extra invalidations per episode against
   thirty-six for walls, and removes the staleness entirely. Named
   `topology_version` rather than `wall_state_version` so the wider meaning
   is visible at the call site.

KNOWN AND ACCEPTED STALENESS. Partial wall damage lowers `wall_tile_cost`
continuously, but the version bumps only on death. A path cached while a wall
was at full HP therefore stays in use while that wall is being chewed down,
even though breaking through has become cheaper. This is deliberate:
re-keying on every HP change would destroy the hit rate, which is the only
reason the cache exists. The error is bounded - it disappears the moment the
wall dies and the version bumps - and it never makes a path invalid, only
mildly sub-optimal.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core.entities import Building
from gymnasium_base_siege.core.generator import BaseSpec

Tile = tuple[int, int]

_NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ===========================================================================
# Cost model
# ===========================================================================


def open_tile_cost(troop_type: str, troop_stats=None) -> float:
    """Ticks to cross one open tile. The unit everything else is measured in."""
    return 1.0 / (C.TROOP_STATS if troop_stats is None else troop_stats)[troop_type]["speed"]


def wall_tile_cost(troop_type: str, wall_hp: int, troop_stats=None) -> float:
    """
    Ticks to break one wall and step onto its tile.

    A troop that cannot damage walls is given an infinite cost rather than a
    division by zero. Nothing in the current stat table hits that branch; it
    exists so a future zero-damage support unit fails by refusing to path
    through walls rather than by crashing mid-episode.
    """
    dps = (C.TROOP_STATS if troop_stats is None else troop_stats)[troop_type]["damage_per_tick"]
    if dps <= 0:
        return float("inf")
    return open_tile_cost(troop_type, troop_stats) + (wall_hp / dps)


def break_even_detour(troop_type: str, wall_hp: int = None) -> float:
    """
    How many EXTRA tiles of walking a troop will accept rather than break one
    wall: speed * wall_hp / damage_per_tick.

    Not used by the search - it is the closed form the search's behaviour is
    checked against, and the source of the D9 figures quoted in the report.
    Keeping it here means those figures are computed from the stat table
    rather than typed into a document and left to rot.
    """
    if wall_hp is None:
        wall_hp = C.BUILDING_STATS[C.WALL]["hp"]
    stats = C.TROOP_STATS[troop_type]
    return stats["speed"] * wall_hp / stats["damage_per_tick"]


# ===========================================================================
# GridState
# ===========================================================================


@dataclass
class GridState:
    """
    The mutable board: every live building, including walls, keyed by tile.

    Owned by combat.py from Stage 5 onward; defined here because pathfinding
    is its first real consumer and an interface written for a real caller is
    a better interface than one written in advance.

    `topology_version` is the cache key ingredient. It changes only when the
    set of passable tiles changes - that is, when a building dies - so a
    cached path is valid for as long as the board's shape is unchanged.
    """

    buildings: dict[Tile, Building] = field(default_factory=dict)
    topology_version: int = 0
    troop_stats: dict | None = field(default=None, repr=False)

    @classmethod
    def from_spec(cls, spec: BaseSpec, *, troop_stats=None, building_stats=None) -> "GridState":
        """Build a fresh board at full health from a BaseSpec."""
        buildings: dict[Tile, Building] = {}
        for btype, row, col in spec.buildings:
            buildings[(row, col)] = Building.spawn(btype, (row, col),
                stats=None if building_stats is None else building_stats[btype])
        for row, col in spec.walls:
            buildings[(row, col)] = Building.spawn(C.WALL, (row, col),
                stats=None if building_stats is None else building_stats[C.WALL])
        return cls(buildings=buildings, troop_stats=troop_stats)

    # -- queries ------------------------------------------------------------

    def building_at(self, tile: Tile) -> Building | None:
        return self.buildings.get(tile)

    def is_wall(self, tile: Tile) -> bool:
        """A live wall: passable, at a cost."""
        building = self.buildings.get(tile)
        return building is not None and building.is_wall and building.is_alive

    def is_blocked(self, tile: Tile) -> bool:
        """A live non-wall building: not passable at any price."""
        building = self.buildings.get(tile)
        return (
            building is not None and not building.is_wall and building.is_alive
        )

    def in_bounds(self, tile: Tile) -> bool:
        row, col = tile
        return 0 <= row < C.GRID_SIZE and 0 <= col < C.GRID_SIZE

    def entry_cost(self, troop_type: str, tile: Tile) -> float:
        """Ticks for `troop_type` to enter `tile`. Infinite if impassable."""
        if not self.in_bounds(tile) or self.is_blocked(tile):
            return float("inf")
        if self.is_wall(tile):
            return wall_tile_cost(troop_type, self.buildings[tile].hp, self.troop_stats)
        return open_tile_cost(troop_type, self.troop_stats)

    # -- mutation -----------------------------------------------------------

    def purge_destroyed(self) -> list[Building]:
        """
        Remove every dead building and bump the version if anything went.

        Called once at the end of a combat tick rather than at each kill, so
        the version advances at most once per tick no matter how many things
        die simultaneously - which matters, because the damage buffer means
        several can die in the same tick.
        """
        dead = [b for tile, b in self.buildings.items() if not b.is_alive]
        if not dead:
            return []
        for building in dead:
            del self.buildings[building.position]
        self.topology_version += 1
        return dead


# ===========================================================================
# PathCache
# ===========================================================================


class PathCache:
    """
    A* with memoisation keyed on (troop_type, start, goal, topology_version).

    Keyed on troop TYPE, not on a troop instance: every melee troop pays the
    same costs, so they all share one entry. With up to 25 troops re-pathing
    across 300 ticks, that sharing is most of the saving.

    The version in the key does the invalidating. When a building dies the
    version advances and every old key becomes unreachable - cheaper and far
    less error-prone than working out which cached paths a particular wall
    actually affected.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, list[Tile] | None] = {}
        self._hits = 0
        self._misses = 0

    # -- statistics ---------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache. Zero before any lookup."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def invalidate(self) -> None:
        """Drop every entry. Statistics are kept - they span the episode."""
        self._cache.clear()

    # -- search -------------------------------------------------------------

    def find_path(
        self,
        troop_type: str,
        start: Tile,
        goal: Tile,
        grid: GridState,
    ) -> list[Tile] | None:
        """
        Cheapest path in ticks from `start` to `goal`, inclusive of both.

        Returns None if no path exists - which, because walls are passable,
        happens only when the goal is itself blocked by a non-wall building or
        is off the board.

        `path[0]` is always `start`, so a caller steps to `path[1]`.
        """
        return self.find_path_to_any(troop_type, start, {goal}, grid)

    def find_path_to_any(
        self,
        troop_type: str,
        start: Tile,
        goals: set[Tile],
        grid: GridState,
    ) -> list[Tile] | None:
        """
        Cheapest path to the nearest member of `goals`.

        Provided for Stage 5. A range-1 troop attacking a building wants "any
        tile adjacent to that building", which is up to four goals; running
        one multi-goal search is strictly better than four single-goal ones,
        and the admissible heuristic is just the minimum over the goal set.
        """
        if not goals:
            return None

        key = (troop_type, start, frozenset(goals), grid.topology_version)
        if key in self._cache:
            self._hits += 1
            cached = self._cache[key]
            return list(cached) if cached is not None else None

        self._misses += 1
        path = self._astar(troop_type, start, goals, grid)
        self._cache[key] = path
        return list(path) if path is not None else None

    # -- the search itself --------------------------------------------------

    def _astar(
        self,
        troop_type: str,
        start: Tile,
        goals: set[Tile],
        grid: GridState,
    ) -> list[Tile] | None:
        if not grid.in_bounds(start):
            return None
        if start in goals:
            return [start]

        step = open_tile_cost(troop_type, grid.troop_stats)

        def heuristic(tile: Tile) -> float:
            row, col = tile
            return step * min(
                abs(row - gr) + abs(col - gc) for gr, gc in goals
            )

        # (f, tile) - the tile is a total order on ints, so ties break
        # deterministically and the same board always yields the same path.
        frontier: list[tuple[float, Tile]] = [(heuristic(start), start)]
        came_from: dict[Tile, Tile] = {}
        best: dict[Tile, float] = {start: 0.0}
        closed: set[Tile] = set()

        while frontier:
            _, current = heapq.heappop(frontier)
            if current in closed:
                continue
            closed.add(current)

            if current in goals:
                return self._reconstruct(came_from, current)

            row, col = current
            for d_row, d_col in _NEIGHBOUR_OFFSETS:
                neighbour = (row + d_row, col + d_col)
                if neighbour in closed:
                    continue

                # The goal tile is entered even if a building sits on it only
                # when it is a wall; a live non-wall building is never a legal
                # destination, which is what keeps paths out of buildings.
                cost = grid.entry_cost(troop_type, neighbour)
                if cost == float("inf"):
                    continue

                tentative = best[current] + cost
                if tentative < best.get(neighbour, float("inf")):
                    best[neighbour] = tentative
                    came_from[neighbour] = current
                    heapq.heappush(
                        frontier, (tentative + heuristic(neighbour), neighbour)
                    )

        return None

    @staticmethod
    def _reconstruct(came_from: dict[Tile, Tile], end: Tile) -> list[Tile]:
        path = [end]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    # -- cost of a path, for tests and diagnostics --------------------------

    @staticmethod
    def path_cost(troop_type: str, path: list[Tile], grid: GridState) -> float:
        """
        Total ticks for a path, excluding the start tile the troop already
        occupies. Used by tests to check that A* really returned the cheapest
        route rather than merely a plausible one.
        """
        return sum(grid.entry_cost(troop_type, tile) for tile in path[1:])
