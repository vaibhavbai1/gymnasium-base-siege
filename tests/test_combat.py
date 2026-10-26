"""
Stage 5 acceptance tests for env/combat.py.

Tests 1-11 are the acceptance criteria from BUILD_ORDER §6 Stage 5.

Test 1 is the most valuable test in the suite and it was written first. Two
entities that would kill each other on the same tick must BOTH die. If damage
is applied inside the loop instead of buffered, whichever unit sits earliest
in the list kills the other and survives - and the result then depends on list
ordering, which is not a game rule.
"""

import random

import pytest

from gymnasium_base_siege.core import combat as B
from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core import generator as G
from gymnasium_base_siege.core import pathfinding as P
from gymnasium_base_siege.core.entities import Building, Troop


# ===========================================================================
# Board builders
# ===========================================================================


def bare_state(buildings=(), troops=(), reserve=None) -> B.BattleState:
    """A board holding only what a test puts on it."""
    grid = P.GridState()
    for btype, tile in buildings:
        grid.buildings[tile] = Building.spawn(btype, tile)

    total = sum(b.value for b in grid.buildings.values() if not b.is_wall)
    state = B.BattleState(grid=grid, total_base_value=max(total, 1))
    state.reserve = dict(reserve or {})

    for troop_type, tile in troops:
        state.troops.append(Troop.spawn(troop_type, tile, state.next_entity_id))
        state.next_entity_id += 1
        state.troops_total += 1

    return state


# ===========================================================================
# BUILD_ORDER Stage 5 acceptance tests
# ===========================================================================


def test_two_entities_that_kill_each_other_both_die():
    """
    Acceptance 1. THE TEST. Damage is buffered and applied simultaneously.

    A melee troop stands adjacent to a cannon. Each is set to exactly the
    other's damage, so one tick is lethal both ways. Troops act in step 4 and
    defences in step 5, so under immediate application the troop would kill
    the cannon first and the dead cannon would never fire.
    """
    state = bare_state(buildings=[("cannon", (5, 5))], troops=[("melee", (5, 6))])
    cannon = state.grid.building_at((5, 5))
    troop = state.troops[0]

    cannon.hp = troop.damage_per_tick      # 15
    troop.hp = cannon.damage_per_shot      # 5

    B.step_tick(state)

    assert not cannon.is_alive, "cannon survived - damage was applied immediately"
    assert not troop.is_alive, "troop survived - damage was applied immediately"


def test_shuffling_the_troop_list_does_not_change_the_outcome():
    """
    Acceptance 2. Order independence, end to end, over 20 shuffles.

    This is the test that catches a non-deterministic tie-break, and it is
    also the one that fails intermittently if you get it wrong - which is far
    worse than failing every time.
    """
    spec = G.generate_base(31, "closed_ring")
    army = [("tank", (0, 4)), ("melee", (0, 5)), ("melee", (0, 6)),
            ("ranged", (0, 7)), ("breaker", (0, 8)), ("breaker", (15, 8)),
            ("melee", (15, 7)), ("ranged", (4, 0))]

    outcomes = set()
    for run in range(20):
        state = B.BattleState.from_spec(spec)
        order = list(army)
        random.Random(run).shuffle(order)
        for troop_type, tile in order:
            state.troops.append(
                Troop.spawn(troop_type, tile, state.next_entity_id)
            )
            state.next_entity_id += 1

        # Ids follow deployment order, so re-key by tile to compare like for
        # like: the question is whether LIST order matters, not id assignment.
        for troop in state.troops:
            troop.entity_id = army.index((troop.troop_type, troop.position)) + 1

        B.run_battle(state)
        outcomes.add((state.destroyed_value, state.tick, state.finish_reason))

    assert len(outcomes) == 1, f"outcome depended on list order: {outcomes}"


def test_mortar_does_not_fire_into_its_blind_spot():
    """Acceptance 3. Distance 1 and 1.41 are both inside min_range 2."""
    for tile in ((5, 6), (6, 6)):          # distance 1.0 and 1.41
        state = bare_state(buildings=[("mortar", (5, 5))], troops=[("melee", tile)])
        troop = state.troops[0]
        B.step_tick(state)
        assert troop.hp == troop.max_hp, f"mortar hit a troop at {tile}"


def test_mortar_fires_at_exactly_its_minimum_range():
    """Acceptance 4. The boundary is inclusive: d >= min_range, not d > min_range."""
    state = bare_state(buildings=[("mortar", (5, 5))], troops=[("melee", (5, 7))])
    troop = state.troops[0]
    assert B.euclidean((5, 5), (5, 7)) == 2.0

    B.step_tick(state)
    assert troop.hp == troop.max_hp - C.BUILDING_STATS["mortar"]["damage_per_shot"]


def test_mortar_splash_falls_off_outside_the_impact_tile():
    """
    Acceptance 5, revised. Radius 1.5 is EUCLIDEAN, so a diagonal at 1.41 is
    inside the blast and a straight neighbour-of-neighbour at 2.0 is outside.
    Manhattan distance here would reach only the four orthogonal tiles.

    The tile the shell lands on takes the full shot; the eight around it take
    SPLASH_FALLOFF of it. Uniform splash made the mortar's aim irrelevant.

    Note (10,5) and (10,6): both are further from the mortar than its own
    range of 4.5, yet both take splash. That is correct - the blast radiates
    from where the shell lands, not from the gun.
    """
    mortar_tile, impact = (5, 5), (9, 5)
    ring = [(9, 4), (9, 6), (10, 4), (10, 5), (10, 6)]   # <= 1.5 from impact
    clear = [(11, 5), (9, 7)]                            # exactly 2.0 away

    state = bare_state(
        buildings=[("mortar", mortar_tile)],
        troops=[("melee", tile) for tile in [impact] + ring + clear],
    )
    mortar = state.grid.building_at(mortar_tile)

    in_band = [
        t for t in state.troops
        if mortar.min_range <= B.euclidean(mortar_tile, t.position) <= mortar.attack_range
    ]
    nearest = min(in_band, key=lambda t: B.euclidean(mortar_tile, t.position))
    assert nearest.position == impact, "the shell does not land where the test assumes"

    B.step_tick(state)

    full = C.BUILDING_STATS["mortar"]["damage_per_shot"]
    reduced = max(1, round(full * C.SPLASH_FALLOFF))
    assert reduced < full, "falloff is not actually reducing anything"

    for troop in state.troops:
        distance = B.euclidean(impact, troop.position)
        if distance == 0:
            expected = full
        elif distance <= 1.5:
            expected = reduced
        else:
            expected = 0
        assert troop.hp == troop.max_hp - expected, (troop.position, distance)


def test_mortar_fires_once_every_cooldown_ticks():
    """
    The mortar is artillery, not a beam. It must be silent between shells, or
    there is no window an approach can be timed into.
    """
    state = bare_state(buildings=[("mortar", (5, 5))], troops=[("tank", (5, 8))])
    tank = state.troops[0]
    cooldown = C.BUILDING_STATS["mortar"]["attack_cooldown"]
    assert cooldown > 1, "the mortar has no cooldown - this test proves nothing"

    shots = []
    previous = tank.hp
    for _ in range(cooldown * 3):
        B.step_tick(state)
        shots.append(tank.hp < previous)
        previous = tank.hp

    fired_on = [i for i, hit in enumerate(shots) if hit]
    assert fired_on == [0, cooldown, cooldown * 2], fired_on


def test_the_cannon_still_fires_every_tick():
    """The cooldown is a mortar change. The cannon's rate must not have moved."""
    state = bare_state(buildings=[("cannon", (5, 5))], troops=[("tank", (5, 7))])
    tank = state.troops[0]
    assert C.BUILDING_STATS["cannon"]["attack_cooldown"] == 1

    for tick in range(1, 6):
        B.step_tick(state)
        expected = tank.max_hp - tick * C.BUILDING_STATS["cannon"]["damage_per_shot"]
        assert tank.hp == expected, tick


def test_a_troop_may_stand_outside_the_mortars_shortened_range():
    """
    Range 5.0 with a blind spot at 2 leaves a band 3 tiles wide. A troop at 6
    tiles is simply safe from that mortar, which was not true at range 6.
    """
    state = bare_state(buildings=[("mortar", (5, 5))], troops=[("tank", (5, 11))])
    tank = state.troops[0]
    assert B.euclidean((5, 5), (5, 11)) == 6.0 > C.BUILDING_STATS["mortar"]["range"]

    B.step_tick(state)
    assert tank.hp == tank.max_hp


def test_tank_targets_a_defence_over_a_nearer_storage():
    """Acceptance 6. Tank targeting is 'nearest defence', not 'nearest building'."""
    state = bare_state(
        buildings=[("storage", (5, 6)), ("cannon", (5, 9))],
        troops=[("tank", (5, 5))],
    )
    tank = state.troops[0]
    B.step_tick(state)

    assert tank.target == (5, 9), "tank went for the storage"
    assert state.grid.building_at((5, 6)).hp == 800


def test_empty_target_pool_falls_back_for_a_tank_but_not_a_breaker():
    """
    Acceptance 6b. D15. A tank with no defences left attacks something else;
    a breaker with no walls left waits, because walls_only means it cannot
    damage a building at all and idling is the honest cost of overbuying.
    """
    tank_state = bare_state(
        buildings=[("storage", (5, 6))], troops=[("tank", (5, 5))]
    )
    B.step_tick(tank_state)
    assert tank_state.troops[0].target == (5, 6)
    assert tank_state.grid.building_at((5, 6)).hp < 800

    breaker_state = bare_state(
        buildings=[("storage", (5, 6))], troops=[("breaker", (5, 5))]
    )
    B.step_tick(breaker_state)
    assert breaker_state.troops[0].target is None, "breaker fell back to a building"
    assert breaker_state.grid.building_at((5, 6)).hp == 800
    assert breaker_state.troops[0].is_alive


def test_breaker_and_wall_die_together_on_one_strike():
    """Acceptance 7. 350 damage against 300 wall HP: one strike, one wall."""
    state = bare_state(buildings=[("wall", (5, 6))], troops=[("breaker", (5, 5))])
    breaker, wall = state.troops[0], state.grid.building_at((5, 6))

    B.step_tick(state)

    assert not wall.is_alive
    assert not breaker.is_alive
    assert C.TROOP_STATS["breaker"]["damage_per_tick"] > C.BUILDING_STATS["wall"]["hp"]


def test_the_standoff_annulus_between_ranged_and_cannon():
    """
    Acceptance 8. A ranged troop reaches 4; a cannon only 3. That one-tile gap
    is the only reason to ever buy a ranged troop - remove it and the unit is
    a strictly worse melee troop.
    """
    state = bare_state(
        buildings=[("cannon", (5, 5))], troops=[("ranged", (5, 9))]
    )
    ranged, cannon = state.troops[0], state.grid.building_at((5, 5))
    assert B.euclidean((5, 5), (5, 9)) == 4.0

    B.step_tick(state)

    assert cannon.hp < cannon.max_hp, "ranged troop could not reach at 4"
    assert ranged.hp == ranged.max_hp, "cannon replied from outside its range"


def test_ranged_hits_across_a_wall_and_melee_cannot():
    """
    Acceptance 9. D7 - nothing blocks a shot, so the ONLY asymmetry is range.
    Both troops stand on the same tile; only one can reach.
    """
    walls = [("wall", (5, 6)), ("wall", (4, 6)), ("wall", (6, 6))]
    storage = ("storage", (5, 8))

    ranged_state = bare_state(buildings=walls + [storage], troops=[("ranged", (5, 5))])
    B.step_tick(ranged_state)
    assert ranged_state.grid.building_at((5, 8)).hp < 800, "ranged could not shoot over"

    melee_state = bare_state(buildings=walls + [storage], troops=[("melee", (5, 5))])
    B.step_tick(melee_state)
    assert melee_state.grid.building_at((5, 8)).hp == 800, "melee shot through a wall"


def test_ties_break_by_ascending_entity_id_for_defences():
    """
    Acceptance 10. O1. Two troops equidistant from a cannon: the lower id is
    hit, every time. Troops can share a tile, so position is not a total order
    for them and the id is what makes this deterministic.
    """
    for order in ((3, 7), (7, 3)):
        state = bare_state(buildings=[("cannon", (5, 5))])
        for entity_id, tile in zip(order, ((5, 7), (5, 3))):
            state.troops.append(Troop.spawn("melee", tile, entity_id))

        B.step_tick(state)
        hurt = [t for t in state.troops if t.hp < t.max_hp]
        assert len(hurt) == 1
        assert hurt[0].entity_id == min(order)


def test_ties_break_by_ascending_position_for_buildings():
    """
    Acceptance 10. O2. Buildings have no entity id by design - two of them can
    never share a tile, so position already identifies one uniquely.
    """
    state = bare_state(
        buildings=[("storage", (5, 3)), ("storage", (3, 5))],
        troops=[("melee", (5, 5))],
    )
    B.step_tick(state)
    assert state.troops[0].target == (3, 5), "did not prefer the lower (row, col)"


def test_a_troop_with_no_target_waits_rather_than_raising():
    """Acceptance 11. A degenerate board must not crash the simulator."""
    state = bare_state(troops=[("melee", (5, 5)), ("breaker", (6, 6))])
    for _ in range(5):
        B.step_tick(state)

    assert all(t.is_alive for t in state.troops)
    assert all(t.target is None for t in state.troops)
    assert state.finish_reason == "full_clear"      # nothing left to destroy


# ===========================================================================
# Guards on invariants that fail silently
# ===========================================================================


def test_the_seven_step_order_is_what_the_module_runs():
    """
    The tick order is a viva question, so it is pinned rather than trusted to
    stay in the docstring.
    """
    import inspect

    source = inspect.getsource(B.step_tick)
    order = [
        "_step_1_remove_dead",
        "_step_2_invalidate_paths",
        "_step_3_retarget",
        "_step_4_troops_act",
        "_step_5_defences_act",
        "_step_6_apply_damage",
        "_step_7_check_end",
    ]
    positions = [source.index(name) for name in order]
    assert positions == sorted(positions), "the tick steps are out of order"


def test_nothing_changes_hit_points_before_step_six():
    """
    Steps 1-5 may only READ the board. If any of them writes a hit point, the
    simulation stops being order-independent whatever step 6 does.
    """
    state = bare_state(
        buildings=[("cannon", (5, 5)), ("wall", (5, 6))],
        troops=[("melee", (5, 7)), ("breaker", (5, 4))],
    )
    before = {id(b): b.hp for b in state.buildings}
    before.update({id(t): t.hp for t in state.troops})

    buffer = B.DamageBuffer()
    attacks = []
    B._step_1_remove_dead(state)
    B._step_2_invalidate_paths(state)
    B._step_3_retarget(state)
    B._step_4_troops_act(state, buffer, attacks)
    B._step_5_defences_act(state, buffer, attacks)

    after = {id(b): b.hp for b in state.buildings}
    after.update({id(t): t.hp for t in state.troops})
    assert after == before, "something took damage before step 6"
    assert len(buffer) > 0, "nothing was buffered - the test proves nothing"

    B._step_6_apply_damage(state, buffer)
    assert any(after[k] != v for k, v in
               {**{id(b): b.hp for b in state.buildings},
                **{id(t): t.hp for t in state.troops}}.items() if k in after)


def test_walls_score_zero_and_scoring_counts_once():
    """
    Destroying 36 walls must not move the score at all, and a building must
    not be scored twice by a second sweep.
    """
    spec = G.generate_base(8, "closed_ring")
    state = B.BattleState.from_spec(spec)
    assert state.total_base_value == 155

    for tile in spec.walls:
        state.grid.building_at(tile).take_damage(10 ** 6)
    B._step_1_remove_dead(state)
    assert state.destroyed_value == 0

    hall = next((r, c) for b, r, c in spec.buildings if b == "town_hall")
    state.grid.building_at(hall).take_damage(10 ** 6)
    B._step_1_remove_dead(state)
    B._step_1_remove_dead(state)
    assert state.destroyed_value == 30
    assert state.milestones["town_hall"] is True


def test_the_tick_limit_terminates_and_names_its_reason():
    """
    The 300-tick deadline is a real ending of the task, not an artificial cut.
    Stage 7 must not bootstrap the value function past it.
    """
    spec = G.generate_base(9, "closed_ring")
    state = B.BattleState.from_spec(spec, reserve={"melee": 5})

    B.run_battle(state)
    assert state.tick == C.BATTLE_TICK_LIMIT
    assert state.finish_reason == "tick_limit"


def test_army_elimination_ends_the_battle():
    state = bare_state(buildings=[("cannon", (5, 5))], troops=[("melee", (5, 6))])
    state.troops[0].hp = 1
    B.run_battle(state)
    assert state.finish_reason == "army_eliminated"


def test_troops_move_at_their_configured_speed():
    """
    A melee troop at 0.16 crosses one tile every 6.25 ticks. It must walk the
    14 tiles from (5,0) to (5,14) - adjacent to its target - in about 88
    ticks, then stop and attack. A tile per tick would put it there in 14 and
    make every balance figure in the report wrong.
    """
    state = bare_state(buildings=[("storage", (5, 15))], troops=[("melee", (5, 0))])
    troop, storage = state.troops[0], state.grid.building_at((5, 15))

    for _ in range(80):
        B.step_tick(state)
    assert troop.position != (5, 14), "arrived too early - moving faster than 0.16"
    assert storage.hp == storage.max_hp

    for _ in range(20):
        B.step_tick(state)
    assert troop.position == (5, 14)
    assert storage.hp < storage.max_hp


def test_deploy_troop_enforces_reserve_and_deployment_tiles():
    state = bare_state(reserve={"melee": 1})
    with pytest.raises(ValueError):
        B.deploy_troop(state, "melee", (8, 8))         # not on the perimeter
    with pytest.raises(ValueError):
        B.deploy_troop(state, "tank", (0, 0))          # none in reserve

    B.deploy_troop(state, "melee", (0, 0))
    assert state.reserve["melee"] == 0
    assert state.troops_total == 1
    assert state.troops[0].entity_id == 1

    with pytest.raises(ValueError):
        B.deploy_troop(state, "melee", (0, 1))


def test_entity_ids_are_never_reused_after_a_death():
    """Ids must stay a stable total order even as troops die."""
    state = bare_state(reserve={"melee": 3})
    B.deploy_troop(state, "melee", (0, 0))
    B.deploy_troop(state, "melee", (0, 1))
    state.troops[0].take_damage(10 ** 6)
    B._step_1_remove_dead(state)
    B.deploy_troop(state, "melee", (0, 2))

    assert [t.entity_id for t in state.troops] == [2, 3]


def test_snapshot_matches_the_render_schema():
    """
    record_snapshot must be a drop-in replacement for render.example_snapshot.
    If it is not, the schema drifted and Stage 14's replay breaks.
    """
    from gymnasium_base_siege.core import render as R

    spec = G.generate_base(12, "ns_gates")
    state = B.BattleState.from_spec(spec, reserve={"melee": 2})
    state.troops.append(Troop.spawn("tank", (0, 0), 1))
    for _ in range(40):
        B.step_tick(state)

    snapshot = B.record_snapshot(state)
    reference = R.example_snapshot()

    assert set(snapshot) == set(reference)
    for key in ("tick", "spent", "troops_alive", "troops_total"):
        assert isinstance(snapshot[key], int)
    assert isinstance(snapshot["destroyed_fraction"], float)
    assert set(snapshot["reserve"]) == set(C.TROOP_TYPES)

    for entry in snapshot["buildings"]:
        assert set(entry) == {"type", "row", "col", "hp", "max_hp"}
    for entry in snapshot["troops"]:
        assert set(entry) == {"type", "row", "col", "hp", "max_hp", "entity_id", "path"}
    for entry in snapshot["attacks"]:
        assert {"from", "to", "kind", "damage"} <= set(entry)
        if entry["kind"] == "defence":
            assert {"impact", "attacker_type", "target_id", "splash"} <= set(entry)
        assert entry["kind"] in ("troop", "defence")


def test_snapshot_is_json_serialisable():
    """A whole episode must be writable to disk and replayable without the simulator."""
    import json

    spec = G.generate_base(13, "layered_core")
    state = B.BattleState.from_spec(spec)
    state.troops.append(Troop.spawn("ranged", (0, 0), 1))
    frames = []
    for _ in range(30):
        B.step_tick(state)
        frames.append(B.record_snapshot(state))

    assert json.loads(json.dumps(frames)) == frames


def test_the_renderer_accepts_a_real_snapshot():
    """
    The payoff for writing render.py first: a real frame must draw without a
    single change to the renderer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gymnasium_base_siege.core import render as R

    spec = G.generate_base(14, "closed_ring")
    state = B.BattleState.from_spec(spec)
    for tile, troop_type in (((0, 6), "tank"), ((0, 7), "breaker"), ((15, 8), "ranged")):
        state.troops.append(Troop.spawn(troop_type, tile, state.next_entity_id))
        state.next_entity_id += 1
    for _ in range(60):
        B.step_tick(state)

    fig, ax = plt.subplots()
    R.draw_snapshot(B.record_snapshot(state), ax, show_ranges=True, show_paths=True)
    plt.close(fig)


def test_a_full_episode_runs_to_a_named_ending():
    """Smoke test across every motif: no crash, always a reason, bounded score."""
    for motif in C.MOTIFS:
        spec = G.generate_base(21, motif)
        state = B.BattleState.from_spec(spec)
        for index, (troop_type, tile) in enumerate(
            [("breaker", (0, 7)), ("tank", (0, 8)), ("melee", (0, 9)),
             ("ranged", (0, 6)), ("melee", (15, 7))]
        ):
            state.troops.append(Troop.spawn(troop_type, tile, index + 1))
            state.next_entity_id = index + 2

        B.run_battle(state)
        assert state.finish_reason in ("full_clear", "tick_limit", "army_eliminated")
        assert 0.0 <= state.destroyed_fraction <= 1.0
        assert state.tick <= C.BATTLE_TICK_LIMIT


def test_no_attack_ever_exceeds_its_range_band():
    """
    Swept across every motif and many seeds: every recorded attack, troop and
    defence alike, must sit inside [min_range, range] of its attacker.

    Eyeballing rendered attack lines is how a range bug survives - an overlong
    line looks like a long line. This checks thousands of them numerically.
    """
    import math

    army = [("breaker", (0, 7)), ("tank", (0, 6)), ("melee", (0, 5)),
            ("ranged", (0, 4)), ("melee", (15, 8))]
    checked = 0

    for motif in C.MOTIFS:
        for seed in (0, 3, 7):
            spec = G.generate_base(seed, motif)
            state = B.BattleState.from_spec(spec)
            for troop_type, tile in army:
                state.troops.append(
                    Troop.spawn(troop_type, tile, state.next_entity_id)
                )
                state.next_entity_id += 1

            while not state.is_finished:
                B.step_tick(state)
                for attack in state.last_attacks:
                    distance = math.dist(attack["from"], attack["to"])
                    checked += 1
                    if attack["kind"] == "defence":
                        stats = C.BUILDING_STATS[attack["attacker_type"]]
                        low, high = stats["min_range"], stats["range"]
                        distance = math.dist(attack["from"], attack["impact"])
                        assert math.dist(attack["impact"], attack["to"]) <= stats["splash_radius"]
                    else:
                        troop_type = next(
                            t for t in C.TROOP_TYPES
                            if C.TROOP_STATS[t]["damage_per_tick"] == attack["damage"]
                        )
                        low, high = 0, C.TROOP_STATS[troop_type]["range"]
                    assert low <= distance <= high + 1e-9, (attack, distance)

    assert checked > 1000, "swept too little to prove anything"
