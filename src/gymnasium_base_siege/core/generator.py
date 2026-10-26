"""
Base Siege - deterministic base generation (structural half).

Section references are to BUILD_ORDER_v2.md §6 Stage 3.

--------------------------------------------------------------------------
WHY THIS MODULE IS THE WAY IT IS
--------------------------------------------------------------------------
Determinism. The same (seed, motif) must always produce a byte-identical
base, because the training / development / final seed split is the backbone
of every claim in the report. If generation is not reproducible, "held-out
layout" means nothing and neither does any number derived from it.

Three habits protect that, and all three are easy to lose by accident:

1. `numpy.random.default_rng(seed)`, never the global `random` module. The
   global module is process-wide state - any import anywhere can reseed it.
2. Sets are SORTED before sampling. Python set iteration order depends on
   insertion order and hash values, so sampling straight from a set gives
   the same seed different bases across runs.
3. `BaseSpec` is frozen with sorted tuple collections, so a base cannot be
   mutated mid-episode and JSON round-trips byte-identically.

Motifs are templates drawn by a loop, never random scatter. A ring drawn by
a loop is always a coherent ring; scattered walls give incoherent bases and
an incoherent difficulty distribution, which would put noise into the very
quantity the experiment is trying to measure.

--------------------------------------------------------------------------
DENSITY IS HELD CONSTANT ON PURPOSE
--------------------------------------------------------------------------
Ring motifs offer 64 placement tiles; the layered core offers 65. Ten
buildings go into either. That near-equality is deliberate: if the transfer
motif also changed how tightly buildings were packed, a transfer result
would confound topology with crowding, and there would be no way to say
which one the agent had failed to generalise over.

This is why the layered core's outer region is bounded at rows/cols 4-12
even though its south and east sides have no wall. Geometrically the region
could run to the buildable limit of 14; it is clipped so the tile count
matches the ring motifs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from gymnasium_base_siege.core import config as C

Tile = tuple[int, int]


# ===========================================================================
# Geometry helpers
# ===========================================================================


def _box_boundary(box: tuple[int, int, int, int]) -> set[Tile]:
    """The perimeter tiles of an inclusive (row_min, col_min, row_max, col_max)."""
    r0, c0, r1, c1 = box
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _box_interior(box: tuple[int, int, int, int]) -> set[Tile]:
    """Everything strictly inside an inclusive box."""
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _box_filled(box: tuple[int, int, int, int]) -> set[Tile]:
    """Every tile of an inclusive box, boundary included."""
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _gate_span(lo: int, hi: int) -> range:
    """
    The centred GATE_WIDTH indices of an inclusive span.

    Corners are never removed (D4). For the 10-tile span 3..12 this returns
    7 and 8 - five tiles clear of either corner - so a gate creates a spatial
    gradient across the edge rather than a funnel onto one tile.
    """
    start = lo + (hi - lo + 1 - C.GATE_WIDTH) // 2
    return range(start, start + C.GATE_WIDTH)


def _edge_tiles(box: tuple[int, int, int, int], side: str) -> set[Tile]:
    """One full edge of an inclusive box, corners included."""
    r0, c0, r1, c1 = box
    if side == "north":
        return {(r0, c) for c in range(c0, c1 + 1)}
    if side == "south":
        return {(r1, c) for c in range(c0, c1 + 1)}
    if side == "west":
        return {(r, c0) for r in range(r0, r1 + 1)}
    if side == "east":
        return {(r, c1) for r in range(r0, r1 + 1)}
    raise ValueError(f"unknown side: {side!r}")


# ===========================================================================
# Motif templates
# ===========================================================================


def wall_tiles(motif: str) -> set[Tile]:
    """
    The wall footprint of a motif. Deterministic; no seed involved.

    Counts, all asserted in tests:
        closed_ring    36    boundary of the 10x10 CENTRED_RING_BOX
        ns_gates       32    that ring, minus 2 tiles north and 2 south
        ew_gates       32    that ring, minus 2 tiles west and 2 east
        layered_core   35    L-shaped outer (19) + full core ring (16)
    """
    if motif not in C.MOTIFS:
        raise ValueError(f"unknown motif: {motif!r}; expected one of {C.MOTIFS}")

    if motif == C.TRANSFER_MOTIF:
        outer = set()
        for side in C.LAYERED_OUTER_SIDES:
            outer |= _edge_tiles(C.LAYERED_OUTER_BOX, side)
        return outer | _box_boundary(C.LAYERED_CORE_BOX)

    ring = _box_boundary(C.CENTRED_RING_BOX)
    r0, c0, r1, c1 = C.CENTRED_RING_BOX

    if motif == "closed_ring":
        return ring
    if motif == "ns_gates":
        gate = _gate_span(c0, c1)
        return ring - {(r0, c) for c in gate} - {(r1, c) for c in gate}
    if motif == "ew_gates":
        gate = _gate_span(r0, r1)
        return ring - {(r, c0) for r in gate} - {(r, c1) for r in gate}

    raise AssertionError(f"motif {motif!r} is in C.MOTIFS but has no template")


# ===========================================================================
# Placement regions
# ===========================================================================


def _inner_region(motif: str) -> set[Tile]:
    """
    The innermost enclosed compartment - where the Town Hall goes.

    Satisfies v3 §6.7's "Town Hall inside the innermost enclosed region"
    check, which is vacuous for single-region motifs and load-bearing for the
    layered core.
    """
    if motif == C.TRANSFER_MOTIF:
        return _box_interior(C.LAYERED_CORE_BOX)      # 3x3 = 9 tiles
    return _box_interior(C.CENTRED_RING_BOX)          # 8x8 = 64 tiles


def _outer_region(motif: str) -> set[Tile]:
    """
    The region outside the innermost compartment but inside the base.

    For ring motifs this is the same 64-tile interior as `_inner_region` -
    there is only one compartment - so the two draws in `generate_base` must
    exclude tiles already used. For the layered core it is the 56 tiles
    between the L and the core footprint.
    """
    if motif != C.TRANSFER_MOTIF:
        return _box_interior(C.CENTRED_RING_BOX)

    r0, c0, r1, c1 = C.LAYERED_OUTER_BOX
    # Clipped at r1/c1 to hold density constant - see the module docstring.
    inside_the_L = {
        (r, c) for r in range(r0 + 1, r1 + 1) for c in range(c0 + 1, c1 + 1)
    }
    return inside_the_L - _box_filled(C.LAYERED_CORE_BOX)


def placement_region(motif: str) -> set[Tile]:
    """
    Every tile a building may occupy for this motif.

    64 tiles for the ring motifs, 65 for the layered core. Never includes a
    wall tile, and never leaves the hard buildable bound of rows/cols 1..14.
    """
    return _inner_region(motif) | _outer_region(motif)


def placement_compartments(motif: str) -> tuple[frozenset[Tile], ...]:
    """
    The placement region split into wall-separated connected components,
    smallest first, using 4-neighbour connectivity to match pathfinding.

    Ring motifs give one compartment. The layered core gives two - the 9-tile
    core and the 56-tile outer region - which is the structural claim the
    transfer motif is chosen to make, and the thing v3 §6.7's "every enclosed
    compartment contains at least one building" check needs in order to mean
    anything.
    """
    region = placement_region(motif)
    seen: set[Tile] = set()
    components: list[frozenset[Tile]] = []

    for tile in sorted(region):
        if tile in seen:
            continue
        stack, component = [tile], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            row, col = current
            for neighbour in ((row - 1, col), (row + 1, col),
                              (row, col - 1), (row, col + 1)):
                if neighbour in region and neighbour not in seen:
                    stack.append(neighbour)
        components.append(frozenset(component))

    return tuple(sorted(components, key=lambda comp: (len(comp), min(comp))))


def reachable_without_breaching(motif: str) -> set[Tile]:
    """
    Every tile reachable from outside the grid with walls impassable.

    Used to prove that gates actually open a base and that the layered core's
    inner compartment genuinely requires a breach. Not used by the simulator;
    kept here so the structural claims in the report are computed from the
    same geometry the environment uses, rather than asserted by hand.
    """
    walls = wall_tiles(motif)
    size = C.GRID_SIZE
    start = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r in (0, size - 1) or c in (0, size - 1)) and (r, c) not in walls
    ]

    seen: set[Tile] = set()
    stack = list(start)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        row, col = current
        for nr, nc in ((row - 1, col), (row + 1, col),
                       (row, col - 1), (row, col + 1)):
            if 0 <= nr < size and 0 <= nc < size:
                if (nr, nc) not in walls and (nr, nc) not in seen:
                    stack.append((nr, nc))
    return seen


# ===========================================================================
# BaseSpec
# ===========================================================================


@dataclass(frozen=True)
class BaseSpec:
    """
    One fully specified base. Frozen, with sorted tuple collections.

    Frozen because a base must not be mutated mid-episode by accident -
    damage belongs on Building instances, not on the specification they were
    built from. Sorted because that is what makes the JSON round-trip
    byte-identical, which is what makes a saved pool reload into exactly the
    bases that were evaluated.
    """

    seed: int
    motif: str
    buildings: tuple[tuple[str, int, int], ...]   # (type, row, col)
    walls: tuple[Tile, ...]

    @property
    def total_value(self) -> int:
        """Sum of non-wall values. Must equal config.TOTAL_BASE_VALUE (155)."""
        return sum(C.BUILDING_STATS[btype]["value"] for btype, _, _ in self.buildings)

    @property
    def occupied(self) -> set[Tile]:
        """Every tile holding a building or a wall."""
        return {(r, c) for _, r, c in self.buildings} | set(self.walls)

    def to_json(self) -> str:
        """
        Compact, key-sorted JSON. The separators argument matters: the default
        inserts a space after each delimiter, and any whitespace variation
        breaks the byte-identity property test 1 depends on.
        """
        return json.dumps(
            {
                "seed": self.seed,
                "motif": self.motif,
                "buildings": [[b, r, c] for b, r, c in self.buildings],
                "walls": [[r, c] for r, c in self.walls],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> "BaseSpec":
        """
        Rebuild from JSON. JSON has no tuple type, so lists must be converted
        back explicitly - without this the round-trip returns an object that
        prints identically and compares unequal, which is a miserable hour.
        """
        data = json.loads(text)
        return cls(
            seed=int(data["seed"]),
            motif=data["motif"],
            buildings=tuple((b, int(r), int(c)) for b, r, c in data["buildings"]),
            walls=tuple((int(r), int(c)) for r, c in data["walls"]),
        )


# ===========================================================================
# Generation
# ===========================================================================


def _contents_to_place(motif: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    (inner, outer) building type lists, expanded in a fixed order.

    Order is fixed by config.BUILDING_TYPES rather than by dict iteration so
    that the sequence of draws from the generator is stable across Python
    versions and across any future edit to BUILDING_COUNTS.
    """
    if motif == C.TRANSFER_MOTIF:
        inner = tuple(C.LAYERED_CORE_CONTENTS)
    else:
        inner = ("town_hall",)

    remaining = dict(C.BUILDING_COUNTS)
    for btype in inner:
        remaining[btype] -= 1
        if remaining[btype] < 0:
            raise ValueError(f"{motif}: core wants more {btype} than the base has")

    outer = tuple(
        btype
        for btype in C.BUILDING_TYPES
        if btype != C.WALL
        for _ in range(remaining.get(btype, 0))
    )
    return inner, outer


def _draw(rng: np.random.Generator, region: set[Tile], n: int) -> list[Tile]:
    """
    Sample n distinct tiles from a region, without replacement.

    `sorted(region)` is the load-bearing part. Sampling straight from a set
    makes the draw depend on set iteration order, which is not stable, and
    the failure is silent: bases still look plausible, they are just not the
    bases the seed claims.
    """
    tiles = sorted(region)
    if n > len(tiles):
        raise ValueError(f"cannot place {n} buildings in {len(tiles)} tiles")
    indices = rng.choice(len(tiles), size=n, replace=False)
    return [tiles[i] for i in indices]


def generate_base(seed: int, motif: str, *, allow_final: bool = False) -> BaseSpec:
    """
    Build one base. Deterministic in (seed, motif).

    NOTE ON SEED SCOPE. The RNG is seeded from `seed` alone, as specified by
    BUILD_ORDER. The three ring motifs share an identical 64-tile placement
    region, so `generate_base(42, "closed_ring")` and
    `generate_base(42, "ns_gates")` place their buildings on exactly the same
    tiles and differ only in walls. That is a clean controlled comparison and
    a reduction in pool diversity at the same time; see the note raised at
    Stage 3 handover. Seeding from (seed, motif) instead is a one-line change
    HERE and nowhere else - but it must be settled before Stage 9 freezes any
    pool, because changing it afterwards silently invalidates every pool.
    """
    from gymnasium_base_siege.core.access import ensure_seed_allowed
    seed = ensure_seed_allowed(seed, allow_final=allow_final)
    if motif not in C.MOTIFS:
        raise ValueError(f"unknown motif: {motif!r}; expected one of {C.MOTIFS}")

    rng = np.random.default_rng(seed)
    walls = wall_tiles(motif)
    inner_types, outer_types = _contents_to_place(motif)

    inner_tiles = _draw(rng, _inner_region(motif), len(inner_types))
    placed = list(zip(inner_types, inner_tiles))

    # Subtract what the inner draw used. For ring motifs the two regions are
    # the same 64 tiles, so without this the Town Hall and a storage could
    # land on one tile.
    outer_available = _outer_region(motif) - set(inner_tiles)
    outer_tiles = _draw(rng, outer_available, len(outer_types))
    placed += list(zip(outer_types, outer_tiles))

    buildings = tuple(sorted((btype, r, c) for btype, (r, c) in placed))
    return BaseSpec(
        seed=seed,
        motif=motif,
        buildings=buildings,
        walls=tuple(sorted(walls)),
    )
