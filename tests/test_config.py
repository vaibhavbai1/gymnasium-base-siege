"""
Stage 1 acceptance tests for env/config.py.

The first four tests are the acceptance criteria from BUILD_ORDER §4.
The rest guard design decisions that are easy to break silently later  -
a broken one of these does not crash anything, it just quietly makes the
experiment measure something other than what the report claims.
"""

from gymnasium_base_siege.core import config as C


# ===========================================================================
# BUILD_ORDER §4 acceptance tests
# ===========================================================================


def test_total_base_value_is_155():
    """Acceptance 1. The 155-point objective scale (plan §6.4)."""
    assert C.TOTAL_BASE_VALUE == 155


def test_wall_score_value_is_exactly_zero():
    """
    Acceptance 2. Walls are obstacles, not objectives (plan §6.4).

    Structural, not a reward penalty: 36 walls at 1 point each would
    collectively outvalue the 30-point Town Hall while being the easiest
    thing on the map to destroy, which makes wall farming optimal rather
    than merely tempting.
    """
    assert C.BUILDING_STATS["wall"]["value"] == 0
    assert C.BUILDING_STATS["wall"]["value"] is not None


def test_deploy_tiles_are_60_unique_perimeter_tiles():
    """Acceptance 3."""
    assert len(C.DEPLOY_TILES) == 60
    assert len(set(C.DEPLOY_TILES)) == 60

    edge = {0, C.GRID_SIZE - 1}
    for row, col in C.DEPLOY_TILES:
        assert row in edge or col in edge, f"({row}, {col}) is not on the perimeter"
        assert 0 <= row < C.GRID_SIZE
        assert 0 <= col < C.GRID_SIZE


def test_no_buildable_coordinate_is_a_deployment_tile():
    """
    Acceptance 4.

    A troop must be able to spawn on any of the 60 perimeter tiles, so no
    building may ever occupy one.
    """
    deploy = set(C.DEPLOY_TILES)
    buildable = {
        (r, c)
        for r in range(C.BUILDABLE_MIN, C.BUILDABLE_MAX + 1)
        for c in range(C.BUILDABLE_MIN, C.BUILDABLE_MAX + 1)
    }
    assert buildable, "buildable region is empty"
    assert deploy.isdisjoint(buildable)


# ===========================================================================
# TOTAL_BASE_VALUE must stay derived, not typed
# ===========================================================================


def test_total_base_value_is_computed_from_the_counts_table():
    """
    BUILD_ORDER Stage 1 pitfall: deriving 155 by hand lets it drift when a
    count or a value changes. Recompute it here independently.
    """
    recomputed = sum(
        C.BUILDING_STATS[btype]["value"] * count
        for btype, count in C.BUILDING_COUNTS.items()
    )
    assert C.TOTAL_BASE_VALUE == recomputed


def test_walls_are_excluded_from_the_counts_table():
    """Walls have no fixed count; their number is set by the motif template."""
    assert "wall" not in C.BUILDING_COUNTS


def test_scoring_building_count_is_ten():
    assert C.N_SCORING_BUILDINGS == 10
    assert C.BUILDING_COUNTS == {
        "town_hall": 1, "cannon": 3, "mortar": 2, "storage": 4
    }


# ===========================================================================
# Design invariants - the stat relationships the whole design rests on
# ===========================================================================


def test_ranged_troop_outranges_the_cannon():
    """
    The standoff invariant.

    On raw stats the ranged troop is strictly worse than melee: less HP
    (90 vs 120), less damage (12 vs 15), higher cost (10 vs 8). The only
    thing that justifies buying one is the annulus where it can hit a
    cannon and the cannon cannot hit back.

    If this ever fails, the rational army contains zero ranged troops and
    the army-composition half of the research question collapses.
    """
    assert C.TROOP_STATS["ranged"]["range"] > C.BUILDING_STATS["cannon"]["range"]


def test_mortar_outranges_the_standoff_band():
    """
    The counter to the standoff.

    Mortar range (6) must exceed ranged troop range (4), so the tiles that
    are safe from cannons are still covered by mortars. Without this the
    standoff band is unconditionally safe and placement becomes trivial.
    """
    assert C.BUILDING_STATS["mortar"]["range"] > C.TROOP_STATS["ranged"]["range"]


def test_mortar_has_a_blind_spot():
    """
    Danger must not be monotone in distance (plan glossary, "blind spot").

    A positive minimum range means the safest tile near a mortar is the one
    right next to it, so the agent has to learn a spatial pattern rather
    than a scalar ordering of "how far from the defences am I".
    """
    assert C.BUILDING_STATS["mortar"]["min_range"] > 0
    assert C.BUILDING_STATS["mortar"]["min_range"] < C.BUILDING_STATS["mortar"]["range"]


def test_only_the_mortar_has_a_blind_spot_or_splash():
    """Every other building fires at everything in range, or not at all."""
    for btype, stats in C.BUILDING_STATS.items():
        if btype == "mortar":
            continue
        assert stats["min_range"] == 0, btype
        assert stats["splash_radius"] == 0.0, btype


def test_one_breaker_removes_exactly_one_wall():
    """Breaker damage 350 > wall HP 300, and the breaker dies on striking."""
    assert C.TROOP_STATS["breaker"]["damage_per_tick"] > C.BUILDING_STATS["wall"]["hp"]
    assert C.TROOP_STATS["breaker"]["dies_after_attack"] is True
    assert C.TROOP_STATS["breaker"]["walls_only"] is True


def test_max_army_size_matches_the_budget_at_the_cheapest_troop():
    """
    MAX_ARMY_SIZE is not an independent constraint: 200 / 8 = 25 exactly.
    It binds only for an all-melee army. Recording that here means a future
    change to the budget or the melee cost surfaces the interaction instead
    of quietly making one of the two constants dead.
    """
    assert C.RESOURCE_BUDGET // C.CHEAPEST_TROOP_COST == C.MAX_ARMY_SIZE


# ===========================================================================
# Action space arithmetic
# ===========================================================================


def test_action_space_size_is_246():
    assert C.N_ACTIONS == 246
    assert C.ACTION_FINISH_SHOPPING == 4
    assert C.ACTION_DEPLOY_FIRST == 5
    assert C.ACTION_DEPLOY_LAST == 244
    assert C.ACTION_WAIT == 245


def test_deployment_action_encoding_round_trips():
    """action = 5 + (troop_index * 60) + tile_index, for every pair."""
    seen = set()
    for troop_index in range(len(C.TROOP_TYPES)):
        for tile_index in range(C.N_DEPLOY_TILES):
            action = C.ACTION_DEPLOY_FIRST + troop_index * C.N_DEPLOY_TILES + tile_index
            assert C.ACTION_DEPLOY_FIRST <= action <= C.ACTION_DEPLOY_LAST
            seen.add(action)

            offset = action - C.ACTION_DEPLOY_FIRST
            assert offset // C.N_DEPLOY_TILES == troop_index
            assert offset % C.N_DEPLOY_TILES == tile_index

    assert len(seen) == C.ACTION_DEPLOY_LAST - C.ACTION_DEPLOY_FIRST + 1 == 240


def test_action_ranges_do_not_overlap():
    buy = set(range(C.ACTION_BUY_FIRST, C.ACTION_BUY_LAST + 1))
    deploy = set(range(C.ACTION_DEPLOY_FIRST, C.ACTION_DEPLOY_LAST + 1))
    singles = {C.ACTION_FINISH_SHOPPING, C.ACTION_WAIT}

    assert len(buy) == len(C.TROOP_TYPES)
    assert buy.isdisjoint(deploy)
    assert singles.isdisjoint(buy)
    assert singles.isdisjoint(deploy)
    assert len(buy | deploy | singles) == C.N_ACTIONS


# ===========================================================================
# Deployment ring ordering - pre-work for Stage 12
# ===========================================================================


def test_90_degree_rotation_is_a_cyclic_shift_of_the_tile_index():
    """
    Written now, not at Stage 12, because it validates the ORDERING choice
    made in config while it is still free to change.

    Clockwise-from-a-corner ordering makes a 90 degree rotation a pure
    cyclic shift of +15. A reflection is not a cyclic shift, which is
    exactly why Stage 12 needs an explicit bijection test.
    """
    index_of = {tile: i for i, tile in enumerate(C.DEPLOY_TILES)}
    n = C.GRID_SIZE

    for i, (row, col) in enumerate(C.DEPLOY_TILES):
        rotated = (col, n - 1 - row)          # 90 degrees clockwise
        assert rotated in index_of
        expected = (i + C.ROTATION_INDEX_SHIFT) % C.N_DEPLOY_TILES
        assert index_of[rotated] == expected


def test_four_rotations_return_to_the_identity():
    assert (4 * C.ROTATION_INDEX_SHIFT) % C.N_DEPLOY_TILES == 0


# ===========================================================================
# Motifs and ring geometry
# ===========================================================================


def _ring(box):
    """Boundary tiles of an inclusive (r0, c0, r1, c1) box."""
    r0, c0, r1, c1 = box
    tiles = {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    return tiles | {(r0, c) for c in range(c0, c1 + 1)} \
                 | {(r1, c) for c in range(c0, c1 + 1)}


def _interior(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def test_every_target_rule_has_a_declared_fallback():
    """
    A rule with no entry here is a rule whose empty-pool behaviour nobody
    decided, which is how the tank ended up standing idle for 200 ticks.
    """
    rules = {C.TROOP_STATS[t]["target_rule"] for t in C.TROOP_TYPES}
    assert rules <= set(C.TARGET_RULE_FALLBACK)
    for rule, fallback in C.TARGET_RULE_FALLBACK.items():
        assert fallback is None or fallback in C.TARGET_RULE_FALLBACK


def test_the_tank_falls_back_but_the_breaker_does_not():
    """
    The tank prefers defences and switches to other buildings once they are
    gone. The breaker is walls_only, so it cannot attack a building at all and
    correctly has no fallback - idling is the cost of buying too many.
    """
    assert C.TROOP_STATS["tank"]["target_rule"] == "nearest_defence"
    assert C.TARGET_RULE_FALLBACK["nearest_defence"] == "nearest_building"

    assert C.TROOP_STATS["breaker"]["walls_only"] is True
    assert C.TARGET_RULE_FALLBACK["nearest_wall"] is None

    for troop in C.TROOP_TYPES:
        rule = C.TROOP_STATS[troop]["target_rule"]
        if C.TARGET_RULE_FALLBACK[rule] == "nearest_building":
            assert C.TROOP_STATS[troop]["walls_only"] is False, troop


def test_nothing_blocks_a_shot():
    """
    Records the 10 August decision. Range alone decides what can be hit.

    The rejected alternative (solid tiles block) would seal every defence
    inside a closed ring, unable to fire outward at all.
    """
    assert C.WALLS_BLOCK_LINE_OF_SIGHT is False
    assert C.BUILDINGS_BLOCK_LINE_OF_SIGHT is False


def test_ranged_is_the_only_troop_that_can_attack_across_a_wall():
    """
    The asymmetry the design depends on, and it falls out of the range table
    rather than a special rule. Every other troop fires at range 1, so it must
    stand adjacent and therefore has to get through the wall.

    If a future stat change gives melee or tank range 2, walls stop mattering
    to three quarters of the army and breakers lose their purpose - so this
    assertion is really a guard on the breaker's reason to exist.
    """
    assert C.WALL_CROSSING_TROOPS == ("ranged",)
    for troop in C.TROOP_TYPES:
        if troop != "ranged":
            assert C.TROOP_STATS[troop]["range"] == 1, troop


def test_a_stationary_ranged_troop_reaches_only_the_interior_edge():
    """
    Pins the consequence of no blocking: a ranged troop deployed on the
    perimeter can already hit the outermost ring of building tiles without
    moving, because that ring sits at exactly its range of 4.

    It must NOT be able to reach deeper than that, or placement stops mattering
    and the whole deployment half of the problem collapses.
    """
    import math

    r0, c0, r1, c1 = C.CENTRED_RING_BOX
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    reach = C.TROOP_STATS["ranged"]["range"]

    hittable = {
        tile for tile in interior
        if any(math.hypot(tile[0] - d[0], tile[1] - d[1]) <= reach
               for d in C.DEPLOY_TILES)
    }
    edge = {t for t in interior
            if t[0] in (r0 + 1, r1 - 1) or t[1] in (c0 + 1, c1 - 1)}

    assert hittable == edge
    assert len(hittable) == 28
    assert len(hittable) < len(interior), "the whole base must not be snipeable"


def test_the_cannon_cannot_answer_a_troop_on_the_deployment_ring():
    """
    A cannon on the interior edge sits 4 tiles from the perimeter, outside its
    own range of 3. The mortar is the only answer to perimeter sniping - which
    is why the mortar's range is 6 and not 4.
    """
    r0 = C.CENTRED_RING_BOX[0]
    distance = r0 + 1
    cannon, mortar = C.BUILDING_STATS["cannon"], C.BUILDING_STATS["mortar"]

    assert distance > cannon["range"]
    assert mortar["min_range"] <= distance <= mortar["range"]


def test_centred_ring_box_produces_36_wall_tiles_inside_the_grid():
    r0, c0, r1, c1 = C.CENTRED_RING_BOX
    assert (r1 - r0 + 1) == (c1 - c0 + 1) == 10

    ring = _ring(C.CENTRED_RING_BOX)
    assert len(ring) == 36
    assert ring.isdisjoint(set(C.DEPLOY_TILES)), "a wall would sit on a deployment tile"


def test_layered_core_ring_is_5x5_and_sits_strictly_inside_the_outer_box():
    r0, c0, r1, c1 = C.LAYERED_CORE_BOX
    assert (r1 - r0 + 1) == (c1 - c0 + 1) == 5
    assert len(_ring(C.LAYERED_CORE_BOX)) == 16

    o_r0, o_c0, o_r1, o_c1 = C.LAYERED_OUTER_BOX
    assert o_r0 < r0 and o_c0 < c0 and r1 < o_r1 and c1 < o_c1, \
        "the core must not touch the outer wall, or they merge into one region"


def test_layered_core_outer_wall_covers_two_adjacent_sides():
    """
    The L must leave a corner open. Two OPPOSITE sides would give a corridor,
    not an L, and would not produce the directional structure the transfer test
    depends on.
    """
    sides = set(C.LAYERED_OUTER_SIDES)
    assert len(sides) == 2
    assert sides <= {"north", "south", "east", "west"}
    assert sides not in ({"north", "south"}, {"east", "west"}), "sides are opposite"


def test_layered_core_has_35_walls_and_clears_the_deployment_ring():
    r0, c0, r1, c1 = C.LAYERED_OUTER_BOX
    edges = {
        "north": {(r0, c) for c in range(c0, c1 + 1)},
        "south": {(r1, c) for c in range(c0, c1 + 1)},
        "west": {(r, c0) for r in range(r0, r1 + 1)},
        "east": {(r, c1) for r in range(r0, r1 + 1)},
    }
    outer = set().union(*(edges[s] for s in C.LAYERED_OUTER_SIDES))
    assert len(outer) == 19, "two adjacent 10-tile edges share one corner"

    walls = outer | _ring(C.LAYERED_CORE_BOX)
    assert len(walls) == 35
    assert walls.isdisjoint(set(C.DEPLOY_TILES))


def test_layered_core_contents_are_available_from_the_building_counts():
    """The core cannot ask for buildings the base does not have."""
    from collections import Counter
    wanted = Counter(C.LAYERED_CORE_CONTENTS)
    for btype, n in wanted.items():
        assert C.BUILDING_COUNTS.get(btype, 0) >= n, btype
    assert "town_hall" in C.LAYERED_CORE_CONTENTS


def test_core_locks_enough_value_to_force_a_second_breach():
    """
    O3. Stripping only the outer region must not be enough to score well, or
    the inner ring is decoration and the motif poses no new problem.

    Locked = 60 of 155, so the outer-only ceiling is 61.3% - comfortably below
    the 85% difficulty gate, which means a good score requires getting inside.
    """
    locked = sum(C.BUILDING_STATS[b]["value"] for b in C.LAYERED_CORE_CONTENTS)
    outer_ceiling = (C.TOTAL_BASE_VALUE - locked) / C.TOTAL_BASE_VALUE

    assert locked == 60
    assert outer_ceiling < C.DIFFICULTY_MAX, (
        f"outer region alone reaches {outer_ceiling:.1%}, at or above the "
        f"{C.DIFFICULTY_MAX:.0%} ceiling - the core is not gating anything"
    )
    assert outer_ceiling > C.DIFFICULTY_MIN, "outer region alone is too sparse to learn from"


def test_the_core_mortar_covers_the_breach_but_not_the_interior():
    """
    Why the core holds a mortar rather than a cannon, revised 24 August 2026.

    The earlier rationale was that a mortar cannot defend the compartment it
    sits in - true of the 3x3 interior, and not the whole picture. Every
    breach has to happen at the WALL RING, because a troop must stand adjacent
    to a wall to break it, and the mortar covers all sixteen of those tiles.
    It also shells almost the whole outer region, which a cannon at range 3
    cannot do.

    The trade is explicit: once a troop is inside the 3x3, nothing there fires
    back. Expensive to reach, defenceless once reached.
    """
    import math

    r0, c0, r1, c1 = C.LAYERED_CORE_BOX
    centre = (r0 + (r1 - r0) // 2, c0 + (c1 - c0) // 2)
    mortar = C.BUILDING_STATS["mortar"]

    def covered(tiles):
        return sum(
            1 for t in tiles
            if mortar["min_range"] <= math.dist(centre, t) <= mortar["range"]
        )

    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    wall_ring = [
        (r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    ]

    assert covered(interior) == 0, "the mortar should be blind inside its own core"
    assert covered(wall_ring) == len(wall_ring), "the breach point must be covered"
    assert "mortar" in C.LAYERED_CORE_CONTENTS
    assert "cannon" not in C.LAYERED_CORE_CONTENTS


def test_the_core_mortar_shells_the_outer_region():
    """
    The reason for the swap. A core cannon reaches 4 outer tiles; a core mortar
    reaches 52 of 56, so the attacker is under artillery for the entire outer
    fight and the only way to stop it is to breach the inner ring. That is a
    tactical problem the three training motifs cannot pose, which is exactly
    what the transfer motif should contribute.
    """
    import math

    r0, c0, r1, c1 = C.LAYERED_CORE_BOX
    centre = (r0 + (r1 - r0) // 2, c0 + (c1 - c0) // 2)
    o0, p0, o1, p1 = C.LAYERED_OUTER_BOX
    outer = [
        (r, c) for r in range(o0 + 1, o1 + 1) for c in range(p0 + 1, p1 + 1)
        if not (r0 <= r <= r1 and c0 <= c <= c1)
    ]

    mortar, cannon = C.BUILDING_STATS["mortar"], C.BUILDING_STATS["cannon"]
    by_mortar = sum(
        1 for t in outer
        if mortar["min_range"] <= math.dist(centre, t) <= mortar["range"]
    )
    by_cannon = sum(1 for t in outer if math.dist(centre, t) <= cannon["range"])

    assert by_mortar > 0.85 * len(outer), by_mortar
    assert by_mortar > 10 * by_cannon, (by_mortar, by_cannon)


def test_layered_core_leaves_room_for_the_town_hall_and_the_rest():
    """
    The core interior must hold the Town Hall; the outer region must hold the
    other nine buildings. Placeable area should also stay close to the closed
    ring's, so building DENSITY is held roughly constant and only topology
    varies between motifs.
    """
    core_interior = _interior(C.LAYERED_CORE_BOX)
    assert len(core_interior) >= len(C.LAYERED_CORE_CONTENTS)

    r0, c0, r1, c1 = C.LAYERED_OUTER_BOX
    cr0, cc0, cr1, cc1 = C.LAYERED_CORE_BOX
    core_footprint = {(r, c) for r in range(cr0, cr1 + 1) for c in range(cc0, cc1 + 1)}
    outer_region = {
        (r, c) for r in range(r0 + 1, r1 + 1) for c in range(c0 + 1, c1 + 1)
    } - core_footprint
    assert len(outer_region) >= C.N_SCORING_BUILDINGS - len(C.LAYERED_CORE_CONTENTS)

    placeable = len(core_interior) + len(outer_region)
    closed_ring_placeable = len(_interior(C.CENTRED_RING_BOX))
    assert abs(placeable - closed_ring_placeable) <= 8, (
        f"{placeable} placeable tiles vs {closed_ring_placeable} for the closed "
        "ring - building density differs too much between motifs"
    )


def test_transfer_motif_is_excluded_from_training():
    """Plan §10.5: the offset ring must never appear in training."""
    assert C.TRANSFER_MOTIF in C.MOTIFS
    assert C.TRANSFER_MOTIF not in C.TRAIN_MOTIFS
    assert set(C.TRAIN_MOTIFS) | {C.TRANSFER_MOTIF} == set(C.MOTIFS)
    assert len(C.MOTIFS) == 4


# ===========================================================================
# Table hygiene
# ===========================================================================


def test_every_troop_type_has_the_same_stat_keys():
    assert set(C.TROOP_STATS) == set(C.TROOP_TYPES)
    key_sets = [frozenset(stats) for stats in C.TROOP_STATS.values()]
    assert len(set(key_sets)) == 1


def test_every_building_type_has_the_same_stat_keys():
    assert set(C.BUILDING_STATS) == set(C.BUILDING_TYPES)
    key_sets = [frozenset(stats) for stats in C.BUILDING_STATS.values()]
    assert len(set(key_sets)) == 1


def test_troop_speeds_are_positive_and_sub_tile():
    """1 / speed is ticks-per-tile, so speed must be > 0 (plan §6.9)."""
    for troop, stats in C.TROOP_STATS.items():
        assert 0 < stats["speed"] <= 1.0, troop


def test_only_defences_deal_damage():
    for btype, stats in C.BUILDING_STATS.items():
        if btype in C.DEFENCE_TYPES:
            assert stats["damage_per_shot"] > 0, btype
            assert stats["range"] > 0, btype
        else:
            assert stats["damage_per_shot"] == 0, btype


# ===========================================================================
# Episode and learning constants
# ===========================================================================


def test_battle_decisions_divide_evenly_into_ticks():
    assert C.BATTLE_TICK_LIMIT % C.TICKS_PER_DECISION == 0
    assert C.MAX_BATTLE_DECISIONS == 30


def test_safety_guard_cannot_fire_before_a_normal_episode_ends():
    """
    The 70-decision guard truncates; everything else terminates (§7.8). It
    must sit clear of a legitimate worst case - up to 25 purchases, one
    finish-shopping, and 30 battle decisions - or it would truncate normal
    play and bootstrap value past the end of the game.
    """
    worst_case = C.MAX_ARMY_SIZE + 1 + C.MAX_BATTLE_DECISIONS   # 56
    assert C.MAX_DECISIONS_PER_EPISODE > worst_case


def test_gamma_is_one():
    """
    Plan §7.7. SB3 discounts per timestep, and shopping does not advance game
    time, so gamma < 1 would penalise a larger army purely for taking more
    decisions to buy.
    """
    assert C.PPO_GAMMA == 1.0


def test_curriculum_stages_sum_to_the_training_budget():
    assert (
        C.CURRICULUM_PLACEMENT_TIMESTEPS + C.CURRICULUM_JOINT_TIMESTEPS
        == C.TOTAL_TRAINING_TIMESTEPS == 262_144
    )


def test_heuristic_army_is_affordable_and_costs_199():
    assert C.HEURISTIC_ARMY_COST == 199
    assert C.HEURISTIC_ARMY_COST <= C.RESOURCE_BUDGET
    assert sum(C.HEURISTIC_ARMY.values()) <= C.MAX_ARMY_SIZE
    assert set(C.HEURISTIC_ARMY) == set(C.TROOP_TYPES)
    assert set(C.HEURISTIC_DEPLOY_ORDER) == set(C.TROOP_TYPES)


def test_reward_milestones_and_penalty_match_the_plan():
    assert C.MILESTONE_HALF_DESTRUCTION == 0.25
    assert C.MILESTONE_TOWN_HALL == 0.25
    assert C.MILESTONE_FULL_CLEAR == 0.25
    assert C.COST_PENALTY_WEIGHT == 0.30
    assert C.HALF_DESTRUCTION_THRESHOLD == 0.50


def test_wall_shaping_bonus_is_off_by_default():
    """
    Plan §6.4 contingency only. If it is ever non-zero, that must be a
    deliberate, written-up decision, and it is excluded from the reported score.
    """
    assert C.WALL_SHAPING_BONUS == 0.0


def test_episode_return_stays_in_a_sane_range_for_the_value_network():
    """
    Plan §7.6: normalising by base value keeps returns near -1 to +2, which
    PPO's value network handles well and thousands badly.
    """
    best_case = 1.0 + C.MILESTONE_HALF_DESTRUCTION + C.MILESTONE_TOWN_HALL \
        + C.MILESTONE_FULL_CLEAR
    worst_case = -C.COST_PENALTY_WEIGHT
    assert -1.0 <= worst_case < best_case <= 2.0


# ===========================================================================
# Seed splits
# ===========================================================================


def test_seed_ranges_do_not_overlap():
    def span(rng):
        return set(range(rng[0], rng[1] + 1))

    train, dev, final = (
        span(C.TRAIN_SEED_RANGE), span(C.DEV_SEED_RANGE), span(C.FINAL_SEED_RANGE)
    )
    assert train.isdisjoint(dev)
    assert train.isdisjoint(final)
    assert dev.isdisjoint(final)


def test_final_study_range_sits_inside_the_protected_range():
    assert C.FINAL_SEED_RANGE[0] <= C.FINAL_STUDY_SEED_RANGE[0]
    assert C.FINAL_STUDY_SEED_RANGE[1] <= C.FINAL_SEED_RANGE[1]
    reserve = C.FINAL_SEED_RANGE[1] - C.FINAL_STUDY_SEED_RANGE[1]
    assert reserve > 0, "no reserve seeds left after the registered final study"


def test_observation_shapes_match_the_plan():
    assert C.N_GRID_LAYERS == 8
    assert C.VECTOR_DIM == 9
    layer_indices = {
        C.GRID_LAYER_TOWN_HALL, C.GRID_LAYER_CANNON, C.GRID_LAYER_MORTAR,
        C.GRID_LAYER_STORAGE, C.GRID_LAYER_WALL, C.GRID_LAYER_BUILDING_HEALTH,
        C.GRID_LAYER_DEFENCE_COVERAGE, C.GRID_LAYER_TROOP_DENSITY,
    }
    assert layer_indices == set(range(C.N_GRID_LAYERS))
    assert C.MAX_DEFENCE_COVERAGE == C.N_DEFENCES == 5
