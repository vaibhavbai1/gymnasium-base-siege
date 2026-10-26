"""
Stage 2 acceptance tests for env/entities.py.

Tests 1-6 are the acceptance criteria from BUILD_ORDER §6 Stage 2. The rest
guard invariants that fail silently rather than loudly: a broken one of these
does not crash the simulator, it just makes it quietly play a different game
from the one the report describes.
"""

import pytest

from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core.entities import Building, Troop


# ===========================================================================
# BUILD_ORDER Stage 2 acceptance tests
# ===========================================================================


def test_slow_troop_advances_exactly_one_tile_after_ten_accumulations():
    """
    Acceptance 1. The whole reason this module exists.

    A tank at speed 0.10 must step after exactly ten ticks - not nine, and
    not never. A float accumulator fails this: 0.10 added ten times is
    0.9999999999999999, which never reaches the threshold.
    """
    tank = Troop.spawn("tank", (0, 0), entity_id=1)
    assert C.TROOP_STATS["tank"]["speed"] == 0.10

    steps = [tank.accumulate_movement() for _ in range(10)]

    assert steps[:9] == [False] * 9, "stepped early"
    assert steps[9] is True, "did not step on the tenth tick"
    assert tank.movement_credit == 0


def test_position_stays_an_integer_tuple():
    """Acceptance 1, second half. Position never becomes a float."""
    tank = Troop.spawn("tank", (3, 4), entity_id=1)
    for _ in range(300):
        tank.accumulate_movement()

    row, col = tank.position
    assert isinstance(row, int) and isinstance(col, int)
    assert tank.position == (3, 4), "accumulate_movement must not move anything"


def test_melee_moves_exactly_sixteen_tiles_in_one_hundred_ticks():
    """
    Acceptance 2. Catches an accumulator that resets to zero instead of
    subtracting one tile. A reset discards 0.12 of progress every tile, so
    this returns 14 rather than 16 and melee quietly arrives late.
    """
    melee = Troop.spawn("melee", (0, 0), entity_id=1)
    assert C.TROOP_STATS["melee"]["speed"] == 0.16

    tiles = sum(melee.accumulate_movement() for _ in range(100))

    assert tiles == 16


def test_every_troop_covers_its_nominal_distance_over_a_full_battle():
    """
    Acceptance 2, generalised. Over the full 300-tick deadline every troop
    must travel speed * 300 tiles exactly - no accumulated shortfall.
    """
    for troop_type in C.TROOP_TYPES:
        troop = Troop.spawn(troop_type, (0, 0), entity_id=1)
        tiles = sum(
            troop.accumulate_movement() for _ in range(C.BATTLE_TICK_LIMIT)
        )
        expected = round(
            C.TROOP_STATS[troop_type]["speed"] * C.BATTLE_TICK_LIMIT
        )
        assert tiles == expected, f"{troop_type} travelled {tiles}, want {expected}"


def test_damage_equal_to_hp_kills():
    """
    Acceptance 3. The boundary at exactly hp == damage is death, not
    survival. An off-by-one here leaves 1-HP survivors all over the board.
    """
    wall = Building.spawn("wall", (5, 5))
    wall.take_damage(wall.hp - 1)
    assert wall.is_alive

    wall.take_damage(1)
    assert not wall.is_alive
    assert wall.hp == 0


def test_take_damage_never_drives_hp_below_zero():
    """
    Acceptance 4. Negative HP puts health_fraction outside Box(0, 1), which
    Gymnasium rejects at Stage 7.
    """
    melee = Troop.spawn("melee", (0, 0), entity_id=1)
    melee.take_damage(999_999)
    assert melee.hp == 0
    assert melee.health_fraction == 0.0

    storage = Building.spawn("storage", (7, 7))
    storage.take_damage(10 ** 9)
    assert storage.hp == 0
    assert 0.0 <= storage.health_fraction <= 1.0


def test_all_nine_entity_types_construct_with_config_stats():
    """Acceptance 5. Table wiring, every type, both classes."""
    for troop_type in C.TROOP_TYPES:
        stats = C.TROOP_STATS[troop_type]
        troop = Troop.spawn(troop_type, (0, 0), entity_id=1)
        assert troop.hp == stats["hp"] == troop.max_hp
        assert troop.damage_per_tick == stats["damage_per_tick"]
        assert troop.attack_range == stats["range"]
        assert troop.speed == stats["speed"]
        assert troop.cost == stats["cost"]
        assert troop.walls_only == stats["walls_only"]
        assert troop.dies_after_attack == stats["dies_after_attack"]
        assert troop.target_rule == stats["target_rule"]

    for building_type in C.BUILDING_TYPES:
        stats = C.BUILDING_STATS[building_type]
        building = Building.spawn(building_type, (8, 8))
        assert building.hp == stats["hp"] == building.max_hp
        assert building.damage_per_shot == stats["damage_per_shot"]
        assert building.attack_cooldown == stats["attack_cooldown"]
        assert building.attack_range == stats["range"]
        assert building.min_range == stats["min_range"]
        assert building.splash_radius == stats["splash_radius"]
        assert building.value == stats["value"]


def test_wall_scores_zero_and_knows_it_is_a_wall():
    """
    Acceptance 6. The scoring rule is a property of the entity, so a caller
    cannot forget to special-case walls.
    """
    wall = Building.spawn("wall", (3, 3))
    assert wall.is_wall is True
    assert wall.value == 0
    assert wall.is_defence is False

    for building_type in C.BUILDING_TYPES:
        if building_type == C.WALL:
            continue
        assert Building.spawn(building_type, (8, 8)).is_wall is False


# ===========================================================================
# Guards on invariants that fail silently
# ===========================================================================


def test_speed_scale_represents_every_speed_exactly():
    """
    If someone adds a three-decimal speed, SPEED_SCALE must go to 1000 or the
    troop silently moves at a rounded speed. Fails here rather than showing
    up as an unexplained balance shift.
    """
    for troop_type in C.TROOP_TYPES:
        scaled = C.TROOP_STATS[troop_type]["speed"] * C.SPEED_SCALE
        assert abs(scaled - round(scaled)) < 1e-9, (
            f"{troop_type} speed is not representable at "
            f"SPEED_SCALE={C.SPEED_SCALE}; raise it to 1000"
        )


def test_movement_credit_is_an_integer_and_stays_bounded():
    """
    No float may enter the movement path. The credit must also stay below one
    full tile after every call, or a step has been missed.
    """
    for troop_type in C.TROOP_TYPES:
        troop = Troop.spawn(troop_type, (0, 0), entity_id=1)
        for _ in range(C.BATTLE_TICK_LIMIT):
            troop.accumulate_movement()
            assert isinstance(troop.movement_credit, int)
            assert 0 <= troop.movement_credit < C.SPEED_SCALE


def test_no_troop_exceeds_one_tile_per_tick():
    """
    accumulate_movement returns a bool, so it can report at most one step.
    That is only sufficient while every speed is below 1.0.
    """
    for troop_type in C.TROOP_TYPES:
        assert C.TROOP_STATS[troop_type]["speed"] < 1.0


def test_accumulation_is_reproducible_across_instances():
    """
    Identical troops must produce identical step sequences. The seed-split
    protocol assumes identical bases give identical episodes.
    """
    for troop_type in C.TROOP_TYPES:
        a = Troop.spawn(troop_type, (0, 0), entity_id=1)
        b = Troop.spawn(troop_type, (9, 9), entity_id=2)
        seq_a = [a.accumulate_movement() for _ in range(300)]
        seq_b = [b.accumulate_movement() for _ in range(300)]
        assert seq_a == seq_b


def test_stats_are_not_copied_onto_instances():
    """
    A stat table change must reach entities that already exist. Copying stats
    in __init__ means a rebalance silently fails to propagate.
    """
    troop = Troop.spawn("melee", (0, 0), entity_id=1)
    original = C.TROOP_STATS["melee"]["damage_per_tick"]
    C.TROOP_STATS["melee"]["damage_per_tick"] = 999
    try:
        assert troop.damage_per_tick == 999
    finally:
        C.TROOP_STATS["melee"]["damage_per_tick"] = original


def test_health_fraction_stays_inside_the_unit_interval():
    """Observation layer 5 is declared Box(0, 1) at Stage 7."""
    cannon = Building.spawn("cannon", (6, 6))
    assert cannon.health_fraction == 1.0
    cannon.take_damage(cannon.max_hp // 2)
    assert 0.0 < cannon.health_fraction < 1.0


def test_negative_damage_does_not_heal():
    """Damage buffering sums many sources; one bad sign must not restore HP."""
    tank = Troop.spawn("tank", (0, 0), entity_id=1)
    tank.take_damage(100)
    before = tank.hp
    tank.take_damage(-50)
    assert tank.hp == before


def test_breaker_damage_still_removes_a_wall_in_one_strike():
    """
    Design invariant D-breaker: one breaker, one wall, then it dies. If a
    rebalance drops breaker damage below wall HP, the troop needs two strikes
    but only survives one, which makes it useless rather than merely weaker.
    """
    breaker = Troop.spawn("breaker", (0, 0), entity_id=1)
    wall = Building.spawn("wall", (3, 3))
    wall.take_damage(breaker.damage_per_tick)
    assert not wall.is_alive
    assert breaker.walls_only is True
    assert breaker.dies_after_attack is True


def test_entity_ids_are_preserved_for_deterministic_tie_breaks():
    """
    Troops can share a tile, so position is not a unique key for them.
    entity_id is what makes "nearest troop" deterministic at Stage 5.
    """
    a = Troop.spawn("melee", (0, 5), entity_id=7)
    b = Troop.spawn("melee", (0, 5), entity_id=3)
    assert a.position == b.position
    assert sorted([a, b], key=lambda t: t.entity_id)[0] is b


def test_unknown_type_fails_loudly():
    """A typo in a type name must raise, not construct a silent zero-stat unit."""
    with pytest.raises(KeyError):
        Troop.spawn("archer", (0, 0), entity_id=1)
    with pytest.raises(KeyError):
        Building.spawn("barracks", (5, 5))
