"""
Stage 4 acceptance tests for env/pathfinding.py.

Tests 1-8 are the acceptance criteria from BUILD_ORDER §6 Stage 4.

Test 3 is the one that matters and it was written first. With mismatched
units - an open tile costing 1 and a wall costing hp/dps - speed cancels out
of the detour comparison entirely, so changing only the speed cannot change
the chosen route and test 3 fails. Every other test here passes under the
buggy model.
"""

import pytest

from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core import generator as G
from gymnasium_base_siege.core import pathfinding as P
from gymnasium_base_siege.core.entities import Building


# ===========================================================================
# Synthetic boards
# ===========================================================================


def make_grid(walls=(), blockers=()) -> P.GridState:
    """A bare board holding only the tiles a test cares about."""
    grid = P.GridState()
    for tile in walls:
        grid.buildings[tile] = Building.spawn(C.WALL, tile)
    for tile in blockers:
        grid.buildings[tile] = Building.spawn("cannon", tile)
    return grid


def detour_board(k: int) -> P.GridState:
    """
    A wall segment in column 5 spanning rows 0..k-1, open at row k.

    From (0, 4) to (0, 6) there are exactly two routes:
        through  two open tiles plus one wall
        around   2k + 2 open tiles

    So the detour costs 2k EXTRA tiles, and going around is cheaper exactly
    when 2k < speed * wall_hp / dps - the closed form in D9.
    """
    return make_grid(walls=[(r, 5) for r in range(k)])


START, GOAL = (0, 4), (0, 6)


@pytest.fixture
def speed_override():
    """Temporarily change one troop's speed, restoring it afterwards."""
    saved = {}

    def _set(troop_type: str, speed: float):
        saved.setdefault(troop_type, C.TROOP_STATS[troop_type]["speed"])
        C.TROOP_STATS[troop_type]["speed"] = speed

    yield _set

    for troop_type, speed in saved.items():
        C.TROOP_STATS[troop_type]["speed"] = speed


# ===========================================================================
# BUILD_ORDER Stage 4 acceptance tests
# ===========================================================================


def test_short_detour_is_preferred_over_breaking_a_wall():
    """
    Acceptance 1. Melee accepts up to 3.2 extra tiles; this board costs 2.
    """
    grid = detour_board(1)
    path = P.PathCache().find_path("melee", START, GOAL, grid)

    assert path is not None
    assert (0, 5) not in path, "broke through a wall it should have walked around"
    assert path[0] == START and path[-1] == GOAL


def test_long_detour_loses_to_breaking_a_wall():
    """
    Acceptance 2. The same board with a 4-tile detour, above melee's 3.2.
    """
    grid = detour_board(2)
    path = P.PathCache().find_path("melee", START, GOAL, grid)

    assert path == [START, (0, 5), GOAL]


def test_changing_only_speed_changes_the_chosen_route(speed_override):
    """
    Acceptance 3. THE UNIT TEST.

    One board, one troop type, one variable: speed. At 0.16 the break-even
    detour is 3.2 tiles and the board's 4-tile detour loses, so melee breaks
    through. At 0.30 the break-even is 6.0 and the same detour wins.

    Under the buggy cost model the comparison is 2k versus hp/dps, in which
    speed does not appear at all, so both cases give the same route and this
    test fails. That is the whole point of it.
    """
    grid = detour_board(2)          # 4 extra tiles

    speed_override("melee", 0.16)   # break-even 3.20 -> break through
    fast_path = P.PathCache().find_path("melee", START, GOAL, grid)

    speed_override("melee", 0.30)   # break-even 6.00 -> walk around
    slow_path = P.PathCache().find_path("melee", START, GOAL, grid)

    assert (0, 5) in fast_path, "should have broken through at speed 0.16"
    assert (0, 5) not in slow_path, "should have detoured at speed 0.30"
    assert fast_path != slow_path


def test_break_even_detour_matches_the_closed_form():
    """
    Acceptance 4. D9, verified rather than asserted. These are the numbers
    quoted in the report, so they are computed from the stat table here.
    """
    expected = {"melee": 3.20, "ranged": 3.50, "tank": 3.75, "breaker": 0.19}
    for troop_type, want in expected.items():
        assert P.break_even_detour(troop_type) == pytest.approx(want, abs=0.005)

    # And the search actually behaves that way at the boundary.
    for troop_type, want in expected.items():
        for k in (1, 2):
            path = P.PathCache().find_path(troop_type, START, GOAL, detour_board(k))
            broke_through = (0, 5) in path
            assert broke_through == (2 * k > want), (troop_type, k, path)


def test_breaker_never_detours():
    """
    Acceptance 4, the design consequence. A break-even of 0.19 tiles means the
    breaker walks through everything - which is what the unit is for.
    """
    assert P.break_even_detour("breaker") < 1.0
    for k in (1, 2, 4, 6):
        path = P.PathCache().find_path("breaker", START, GOAL, detour_board(k))
        assert path == [START, (0, 5), GOAL], k


def test_no_path_ever_enters_a_non_wall_building():
    """Acceptance 5. Buildings block absolutely; walls do not."""
    spec = G.generate_base(7, "closed_ring")
    grid = P.GridState.from_spec(spec)
    cache = P.PathCache()
    occupied = {(r, c) for b, r, c in spec.buildings}

    for start in ((0, 0), (0, 15), (15, 0), (15, 15), (0, 8), (8, 0)):
        for goal in sorted(G.placement_region("closed_ring") - occupied)[:6]:
            path = cache.find_path("melee", start, goal, grid)
            assert path is not None, (start, goal)
            assert not (set(path) & occupied), (start, goal)


def test_cached_lookup_equals_an_uncached_recompute():
    """Acceptance 6. The cache is not lying."""
    grid = P.GridState.from_spec(G.generate_base(11, "ns_gates"))
    warm, cold = P.PathCache(), P.PathCache()

    goals = sorted(G.placement_region("ns_gates"))[:8]
    for goal in goals:
        warm.find_path("tank", (0, 0), goal, grid)

    for goal in goals:
        assert warm.find_path("tank", (0, 0), goal, grid) == cold.find_path(
            "tank", (0, 0), goal, grid
        )

    assert warm.hit_rate == 0.5, warm.hit_rate    # 8 misses then 8 hits
    assert cold.hit_rate == 0.0


def test_wall_death_bumps_the_version_and_changes_the_path():
    """
    Acceptance 7. Invalidation is wired to a building dying, not to time.
    """
    grid = P.GridState.from_spec(G.generate_base(0, "closed_ring"))
    cache = P.PathCache()
    start, goal = (7, 0), (7, 7)

    before = cache.find_path("melee", start, goal, grid)
    assert (7, 3) in before, "direct route should cross the ring wall"

    version_before = grid.topology_version
    grid.building_at((6, 3)).take_damage(10 ** 6)
    destroyed = grid.purge_destroyed()

    assert len(destroyed) == 1
    assert grid.topology_version == version_before + 1

    after = cache.find_path("melee", start, goal, grid)
    assert after != before, "path did not react to the breach"
    assert (6, 3) in after
    assert P.PathCache.path_cost("melee", after, grid) < P.PathCache.path_cost(
        "melee", before, grid
    )


def test_a_fully_sealed_ring_still_yields_a_path():
    """
    Acceptance 8. Walls are expensive, not impassable, so no base is
    structurally unattackable and the generator never has to prove one isn't.
    """
    spec = G.generate_base(3, "closed_ring")
    grid = P.GridState.from_spec(spec)
    occupied = {(r, c) for b, r, c in spec.buildings}
    interior = sorted(G.placement_region("closed_ring") - occupied)

    assert not (G.reachable_without_breaching("closed_ring") & set(interior))

    for troop_type in C.TROOP_TYPES:
        path = P.PathCache().find_path(troop_type, (0, 0), interior[0], grid)
        assert path is not None, troop_type
        assert any(grid.is_wall(tile) for tile in path), "must breach something"


# ===========================================================================
# Guards on invariants that fail silently
# ===========================================================================


def test_costs_are_in_ticks_not_grid_steps():
    """
    The unit itself. An open tile must cost 1/speed, never 1 - a cost of 1.0
    for any troop is the signature of the bug.
    """
    for troop_type in C.TROOP_TYPES:
        speed = C.TROOP_STATS[troop_type]["speed"]
        assert P.open_tile_cost(troop_type) == pytest.approx(1 / speed)
        assert P.open_tile_cost(troop_type) > 1.0

    assert P.open_tile_cost("melee") == pytest.approx(6.25)
    assert P.wall_tile_cost("melee", 300) == pytest.approx(6.25 + 20.0)


def test_a_building_death_also_invalidates():
    """
    Not in the contract, which bumps only on wall death. A destroyed cannon
    opens its tile; without this, every cached path keeps routing around a
    building that is no longer there.
    """
    grid = make_grid(blockers=[(0, 5)])
    cache = P.PathCache()

    before = cache.find_path("melee", START, GOAL, grid)
    assert (0, 5) not in before

    grid.building_at((0, 5)).take_damage(10 ** 6)
    grid.purge_destroyed()

    after = cache.find_path("melee", START, GOAL, grid)
    assert after == [START, (0, 5), GOAL], "cache served a stale detour"


def test_invalidate_clears_entries_but_keeps_statistics():
    grid = detour_board(1)
    cache = P.PathCache()
    cache.find_path("melee", START, GOAL, grid)
    assert cache.size == 1

    cache.invalidate()
    assert cache.size == 0
    assert cache._misses == 1, "statistics span the episode, not the cache"


def test_paths_are_deterministic_and_start_at_the_start():
    """
    Ties in A* must break the same way every time, or two identical episodes
    diverge and the seed protocol stops meaning anything.
    """
    spec = G.generate_base(5, "layered_core")
    grid = P.GridState.from_spec(spec)
    occupied = {(r, c) for b, r, c in spec.buildings}
    goal = sorted(G.placement_region("layered_core") - occupied)[0]

    for troop_type in C.TROOP_TYPES:
        runs = [
            P.PathCache().find_path(troop_type, (0, 0), goal, grid)
            for _ in range(5)
        ]
        assert all(run == runs[0] for run in runs), troop_type
        assert runs[0][0] == (0, 0) and runs[0][-1] == goal


def test_path_steps_are_four_connected():
    """No diagonal slipping between wall corners."""
    grid = P.GridState.from_spec(G.generate_base(2, "ew_gates"))
    path = P.PathCache().find_path("ranged", (0, 0), (9, 9), grid)
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        assert abs(r0 - r1) + abs(c0 - c1) == 1


def test_multi_goal_search_finds_the_nearest_goal():
    """
    Stage 5 needs "any tile adjacent to this building". One multi-goal search
    must return what the cheapest of several single-goal searches would.
    """
    grid = P.GridState.from_spec(G.generate_base(4, "closed_ring"))
    cache = P.PathCache()
    goals = {(5, 5), (9, 9), (11, 4)}

    combined = cache.find_path_to_any("tank", (0, 0), goals, grid)
    singles = [
        P.PathCache().find_path("tank", (0, 0), goal, grid) for goal in goals
    ]
    cheapest = min(P.PathCache.path_cost("tank", p, grid) for p in singles)

    assert combined[-1] in goals
    assert P.PathCache.path_cost("tank", combined, grid) == pytest.approx(cheapest)


def test_goal_on_a_live_building_has_no_path():
    """
    A non-wall building is never a legal destination. Stage 5 must path to an
    adjacent tile, which is what find_path_to_any exists for.
    """
    spec = G.generate_base(6, "closed_ring")
    grid = P.GridState.from_spec(spec)
    hall = next((r, c) for b, r, c in spec.buildings if b == "town_hall")

    assert P.PathCache().find_path("melee", (0, 0), hall, grid) is None

    adjacent = {
        (hall[0] + dr, hall[1] + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
    }
    assert P.PathCache().find_path_to_any("melee", (0, 0), adjacent, grid) is not None


def test_hit_rate_is_zero_before_any_lookup():
    assert P.PathCache().hit_rate == 0.0


def test_grid_state_builds_a_complete_board_from_a_spec():
    for motif in C.MOTIFS:
        spec = G.generate_base(1, motif)
        grid = P.GridState.from_spec(spec)
        assert len(grid.buildings) == len(spec.buildings) + len(spec.walls)
        assert sum(1 for t in grid.buildings if grid.is_wall(t)) == len(spec.walls)
        assert sum(1 for t in grid.buildings if grid.is_blocked(t)) == 10
