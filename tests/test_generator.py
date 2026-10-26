"""
Stage 3 acceptance tests for env/generator.py.

Tests 1-10 are the acceptance criteria from BUILD_ORDER §6 Stage 3. The rest
guard geometry and determinism invariants that fail silently: a base that is
subtly wrong still trains, still evaluates, and still produces a number for
the report.
"""

import pytest

from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core import generator as G


SEEDS = list(range(20))


# ===========================================================================
# BUILD_ORDER Stage 3 acceptance tests
# ===========================================================================


def test_same_seed_and_motif_produce_byte_identical_json():
    """
    Acceptance 1. The foundation of the seed split. If this fails, every
    "held-out layout" claim in the report is unsupported.
    """
    first = G.generate_base(42, "closed_ring")
    second = G.generate_base(42, "closed_ring")
    assert first.to_json() == second.to_json()
    assert first == second


def test_determinism_holds_for_every_motif_and_many_seeds():
    """Acceptance 1, widened. One lucky seed proves nothing."""
    for motif in C.MOTIFS:
        for seed in SEEDS:
            assert (
                G.generate_base(seed, motif).to_json()
                == G.generate_base(seed, motif).to_json()
            )


def test_different_seeds_give_different_building_positions():
    """Acceptance 2. Proves the seed actually reaches the placement draw."""
    for motif in C.MOTIFS:
        layouts = {G.generate_base(seed, motif).buildings for seed in SEEDS}
        assert len(layouts) > 1, f"{motif}: seed has no effect on placement"


def test_every_base_has_ten_buildings_with_the_right_counts():
    """
    Acceptance 3. Firepower is constant across bases, so episode reward
    measures the agent rather than which base it happened to draw.
    """
    for motif in C.MOTIFS:
        for seed in SEEDS:
            spec = G.generate_base(seed, motif)
            assert len(spec.buildings) == C.N_SCORING_BUILDINGS == 10

            counts = {}
            for btype, _, _ in spec.buildings:
                counts[btype] = counts.get(btype, 0) + 1
            assert counts == C.BUILDING_COUNTS, f"{motif}/{seed}: {counts}"


def test_no_building_on_a_wall_and_no_two_on_one_tile():
    """Acceptance 4. Sampling is without replacement and excludes walls."""
    for motif in C.MOTIFS:
        walls = set(G.wall_tiles(motif))
        for seed in SEEDS:
            spec = G.generate_base(seed, motif)
            tiles = [(r, c) for _, r, c in spec.buildings]
            assert len(set(tiles)) == len(tiles), f"{motif}/{seed}: stacked buildings"
            assert not (set(tiles) & walls), f"{motif}/{seed}: building on a wall"


def test_all_buildings_lie_inside_the_placement_region():
    """Acceptance 5. The region rule is enforced, not merely documented."""
    for motif in C.MOTIFS:
        region = G.placement_region(motif)
        for seed in SEEDS:
            spec = G.generate_base(seed, motif)
            for btype, row, col in spec.buildings:
                assert (row, col) in region, f"{motif}/{seed}: {btype} at {(row, col)}"


def test_wall_counts_are_exact_per_motif():
    """Acceptance 6. Templates drawn correctly; gates actually remove tiles."""
    expected = {
        "closed_ring": 36,
        "ns_gates": 32,
        "ew_gates": 32,
        "layered_core": 35,
    }
    for motif, count in expected.items():
        assert len(G.wall_tiles(motif)) == count, motif
        assert len(G.generate_base(0, motif).walls) == count, motif


def test_gates_open_the_base_and_the_closed_ring_does_not():
    """
    Acceptance 7. Reachability from outside the grid with walls impassable  -
    the structural property, not just a tile count.
    """
    interior = G._box_interior(C.CENTRED_RING_BOX)

    assert not (G.reachable_without_breaching("closed_ring") & interior), (
        "closed ring is not closed"
    )
    for motif in ("ns_gates", "ew_gates"):
        assert G.reachable_without_breaching(motif) >= interior, (
            f"{motif}: gate does not actually open the base"
        )


def test_layered_core_has_two_compartments_and_ring_motifs_have_one():
    """
    Acceptance 8. The transfer motif is structurally what the report claims:
    a second compartment nested inside the first.
    """
    for motif in ("closed_ring", "ns_gates", "ew_gates"):
        assert len(G.placement_compartments(motif)) == 1, motif

    compartments = G.placement_compartments(C.TRANSFER_MOTIF)
    assert len(compartments) == 2
    assert sorted(len(comp) for comp in compartments) == [9, 56]


def test_layered_core_inner_compartment_requires_a_breach():
    """
    Acceptance 8, the part that matters. The outer region of the layered core
    is open - its south and east sides have no wall - but the core is not.
    Two nested layers is the claim; this is the test of it.
    """
    reachable = G.reachable_without_breaching(C.TRANSFER_MOTIF)
    outer, core = G._outer_region(C.TRANSFER_MOTIF), G._inner_region(C.TRANSFER_MOTIF)

    assert outer <= reachable, "outer region should be walk-in"
    assert not (core & reachable), "core should require breaking a wall"


def test_total_value_is_155_for_every_motif_and_seed():
    """Acceptance 9. The objective scale never drifts."""
    for motif in C.MOTIFS:
        for seed in SEEDS:
            spec = G.generate_base(seed, motif)
            assert spec.total_value == C.TOTAL_BASE_VALUE == 155


def test_json_round_trip_returns_an_equal_spec():
    """Acceptance 10. Pools reload into exactly the bases that were evaluated."""
    for motif in C.MOTIFS:
        for seed in (0, 7, 4242):
            spec = G.generate_base(seed, motif)
            restored = G.BaseSpec.from_json(spec.to_json())
            assert restored == spec
            assert restored.to_json() == spec.to_json()
            assert isinstance(restored.walls[0], tuple), "lists must become tuples"


# ===========================================================================
# Guards on invariants that fail silently
# ===========================================================================


def test_placement_region_sizes_are_64_and_65():
    """
    Density is held constant so that transfer measures topology, not
    crowding. If these drift apart, a transfer result becomes uninterpretable.
    """
    for motif in ("closed_ring", "ns_gates", "ew_gates"):
        assert len(G.placement_region(motif)) == 64, motif
    assert len(G.placement_region(C.TRANSFER_MOTIF)) == 65


def test_placement_region_never_overlaps_walls():
    for motif in C.MOTIFS:
        assert not (G.placement_region(motif) & G.wall_tiles(motif)), motif


def test_everything_stays_inside_the_hard_buildable_bound():
    """No building or wall may sit on the outermost ring of the grid."""
    for motif in C.MOTIFS:
        tiles = G.placement_region(motif) | G.wall_tiles(motif)
        for row, col in tiles:
            assert C.BUILDABLE_MIN <= row <= C.BUILDABLE_MAX, (motif, row, col)
            assert C.BUILDABLE_MIN <= col <= C.BUILDABLE_MAX, (motif, row, col)


def test_gates_are_centred_and_never_remove_a_corner():
    """
    D4. A centred gate creates a spatial gradient across its edge; a gate on
    a corner would funnel the whole army onto one tile.
    """
    ring = G.wall_tiles("closed_ring")
    r0, c0, r1, c1 = C.CENTRED_RING_BOX
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

    for motif in ("ns_gates", "ew_gates"):
        removed = ring - G.wall_tiles(motif)
        assert len(removed) == 2 * C.GATE_WIDTH
        assert not (removed & corners), f"{motif} removed a corner"

    ns_removed = sorted(ring - G.wall_tiles("ns_gates"))
    assert {c for _, c in ns_removed} == {7, 8}, ns_removed


def test_layered_core_contents_land_in_the_core():
    """
    D14, revised. The core holds a Town Hall, a MORTAR and a storage - 60 of
    155 locked behind the second wall, which caps outer-only destruction at
    61.3% and forces a double breach.
    """
    core = G._inner_region(C.TRANSFER_MOTIF)
    for seed in SEEDS:
        spec = G.generate_base(seed, C.TRANSFER_MOTIF)
        inside = sorted(b for b, r, c in spec.buildings if (r, c) in core)
        assert inside == sorted(C.LAYERED_CORE_CONTENTS), (seed, inside)

    locked = sum(C.BUILDING_STATS[b]["value"] for b in C.LAYERED_CORE_CONTENTS)
    assert locked == 60
    assert (C.TOTAL_BASE_VALUE - locked) / C.TOTAL_BASE_VALUE < 0.85


def test_town_hall_is_in_the_innermost_compartment_for_every_motif():
    """v3 §6.7's structural check, which the layered core makes non-vacuous."""
    for motif in C.MOTIFS:
        innermost = G.placement_compartments(motif)[0]
        for seed in SEEDS:
            spec = G.generate_base(seed, motif)
            hall = next((r, c) for b, r, c in spec.buildings if b == "town_hall")
            assert hall in innermost, (motif, seed, hall)


def test_every_compartment_contains_at_least_one_building():
    """v3 §6.7's second structural check. Vacuous before the layered core."""
    for motif in C.MOTIFS:
        compartments = G.placement_compartments(motif)
        for seed in SEEDS:
            tiles = {(r, c) for _, r, c in G.generate_base(seed, motif).buildings}
            for comp in compartments:
                assert comp & tiles, f"{motif}/{seed}: empty compartment"


def test_collections_are_sorted_so_json_is_stable():
    for motif in C.MOTIFS:
        spec = G.generate_base(3, motif)
        assert list(spec.buildings) == sorted(spec.buildings)
        assert list(spec.walls) == sorted(spec.walls)


def test_basespec_is_frozen():
    """A base must not be mutated mid-episode. Damage belongs on entities."""
    spec = G.generate_base(1, "closed_ring")
    with pytest.raises(Exception):
        spec.seed = 2


def test_unknown_motif_fails_loudly():
    for bad in ("offset_ring", "spiral", ""):
        with pytest.raises(ValueError):
            G.wall_tiles(bad)
        with pytest.raises(ValueError):
            G.generate_base(0, bad)


def test_generation_does_not_touch_the_global_random_module():
    """
    The single most common way to lose reproducibility. If generator ever
    reaches for `random`, the global sequence advances and this catches it.
    """
    import random

    random.seed(12345)
    expected = [random.random() for _ in range(3)]

    random.seed(12345)
    for motif in C.MOTIFS:
        G.generate_base(99, motif)
    assert [random.random() for _ in range(3)] == expected


def test_ring_motifs_share_a_layout_at_a_given_seed():
    """
    Documents a live consequence rather than asserting it is correct: the RNG
    is seeded from `seed` alone and the three ring motifs share a placement
    region, so one seed gives all three the same buildings on the same tiles.

    Clean for paired motif comparison, and a threefold reduction in pool
    diversity. Must be settled before Stage 9 freezes a pool. If the decision
    is to seed from (seed, motif), this test flips to assertNotEqual.
    """
    layouts = {
        motif: G.generate_base(42, motif).buildings
        for motif in ("closed_ring", "ns_gates", "ew_gates")
    }
    assert layouts["closed_ring"] == layouts["ns_gates"] == layouts["ew_gates"]
