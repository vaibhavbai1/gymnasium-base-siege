"""
Stage 5 acceptance tests for env/render.py.

The renderer is written ahead of combat.py because it consumes a snapshot
dict rather than live objects. These tests therefore also serve as the
executable specification of the snapshot schema: when combat.record_snapshot()
lands, feeding its output through draw_snapshot must work unchanged.
"""

import matplotlib
matplotlib.use("Agg")           # headless; must precede the pyplot import

import matplotlib.pyplot as plt
import pytest

from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core import render as R


@pytest.fixture
def snapshot():
    return R.example_snapshot()


@pytest.fixture
def ax():
    fig, axis = plt.subplots()
    yield axis
    plt.close(fig)


# ===========================================================================
# BUILD_ORDER_v2 Stage 5 acceptance tests
# ===========================================================================


@pytest.mark.parametrize("style", ["debug", "presentation"])
def test_draw_snapshot_runs_headless_in_both_styles(snapshot, ax, style):
    """Acceptance 1. No exception, no display required."""
    R.draw_snapshot(snapshot, ax, style=style)
    assert ax.patches, "nothing was drawn"


def test_every_live_entity_appears_on_the_axes(snapshot, ax):
    """
    Acceptance 2. Nothing is silently dropped.

    In debug style each live entity is exactly one rounded patch, so the count
    is checkable. Board furniture (ground, 60 deploy tiles, border) is
    subtracted first.
    """
    R.draw_snapshot(snapshot, ax, style="debug", show_health=False)

    board_patches = 1 + len(C.DEPLOY_TILES) + 1
    live = sum(1 for b in snapshot["buildings"] if b["hp"] > 0)
    live += sum(1 for t in snapshot["troops"] if t["hp"] > 0)

    assert len(ax.patches) - board_patches == live


def test_range_overlay_draws_one_circle_per_defence_plus_blind_spots(snapshot, ax):
    """
    Acceptance 3.

    Every live defence gets an outer circle; only the mortar gets a second,
    inner one at its minimum range. That inner circle is the blind spot  -
    the single most useful thing this overlay shows.
    """
    from matplotlib.patches import Circle

    R.draw_snapshot(snapshot, ax, show_ranges=True, show_health=False)
    circles = [p for p in ax.patches if isinstance(p, Circle)]

    live_defences = [b for b in snapshot["buildings"]
                     if b["hp"] > 0 and C.BUILDING_STATS[b["type"]]["range"] > 0]
    live_mortars = [b for b in live_defences if b["type"] == "mortar"]

    assert len(circles) == len(live_defences) + len(live_mortars)

    radii = sorted(c.radius for c in circles)
    assert C.BUILDING_STATS["mortar"]["min_range"] in radii
    assert C.BUILDING_STATS["mortar"]["range"] in radii
    assert C.BUILDING_STATS["cannon"]["range"] in radii


# ===========================================================================
# Extra guards
# ===========================================================================


def test_destroyed_entities_are_not_drawn(snapshot, ax):
    """A dead building must vanish, not linger at zero health."""
    snapshot["buildings"][-1]["hp"] = 0
    snapshot["troops"][0]["hp"] = 0

    R.draw_snapshot(snapshot, ax, style="debug", show_health=False)

    board_patches = 1 + len(C.DEPLOY_TILES) + 1
    live = sum(1 for b in snapshot["buildings"] if b["hp"] > 0)
    live += sum(1 for t in snapshot["troops"] if t["hp"] > 0)
    assert len(ax.patches) - board_patches == live


def test_health_bars_add_two_patches_per_live_entity(snapshot, ax):
    """Backing plus foreground. Catches a bar that renders but never fills."""
    fig2, bare = plt.subplots()
    R.draw_snapshot(snapshot, bare, style="debug", show_health=False)
    without = len(bare.patches)
    plt.close(fig2)

    R.draw_snapshot(snapshot, ax, style="debug", show_health=True)
    live = sum(1 for b in snapshot["buildings"] if b["hp"] > 0)
    live += sum(1 for t in snapshot["troops"] if t["hp"] > 0)

    assert len(ax.patches) - without == 2 * live


def test_attack_lines_are_drawn_one_per_attack(snapshot, ax):
    """
    Attack lines are the highest-value debugging feature in the module: a tank
    drawing a line to a storage instead of a cannon tells you the targeting
    rule is wrong, instantly.
    """
    from matplotlib.text import Annotation

    R.draw_snapshot(snapshot, ax)
    arrows = [t for t in ax.texts
              if isinstance(t, Annotation) and t.arrow_patch is not None]
    assert len(arrows) == len(snapshot["attacks"])


def test_path_overlay_is_off_by_default_and_skips_troops_without_paths(snapshot, ax):
    fig2, bare = plt.subplots()
    R.draw_snapshot(snapshot, bare, show_paths=False)
    without = len(bare.lines)
    plt.close(fig2)

    R.draw_snapshot(snapshot, ax, show_paths=True)
    with_paths = sum(1 for t in snapshot["troops"]
                     if t.get("path") and len(t["path"]) >= 2)
    assert len(ax.lines) - without == with_paths


def test_damage_overlay_aggregates_per_target_tile(snapshot, ax):
    """
    Splash puts several attacks on one tile in a single tick. They must
    aggregate into one number, not stack unreadably.
    """
    snapshot["attacks"] = [
        {"from": [7, 10], "to": [13, 7], "kind": "defence", "damage": 9},
        {"from": [7, 10], "to": [13, 7], "kind": "defence", "damage": 9},
    ]
    R.draw_snapshot(snapshot, ax, show_damage=True)
    labels = [t.get_text() for t in ax.texts if t.get_text().startswith("-")]
    assert labels == ["-18"]


def test_an_unknown_style_raises(snapshot, ax):
    with pytest.raises(ValueError):
        R.draw_snapshot(snapshot, ax, style="isometric")


def test_every_entity_type_has_a_glyph_and_a_colour():
    """A stat table addition must not silently render as nothing."""
    expected = set(C.BUILDING_TYPES) | set(C.TROOP_TYPES)
    assert set(R.GLYPHS) == expected
    assert set(R.COLOURS) == expected


def test_example_snapshot_matches_the_config_it_claims_to_use(snapshot):
    """
    The fixture is the schema contract for combat.record_snapshot(). If it
    drifts from config, the tests above stop proving anything useful.
    """
    scoring = [b for b in snapshot["buildings"] if b["type"] != "wall"]
    assert len(scoring) == C.N_SCORING_BUILDINGS
    assert sum(C.BUILDING_STATS[b["type"]]["value"] for b in scoring) == 155

    for entity in snapshot["buildings"] + snapshot["troops"]:
        table = (C.BUILDING_STATS if entity["type"] in C.BUILDING_TYPES
                 else C.TROOP_STATS)
        assert entity["max_hp"] == table[entity["type"]]["hp"], entity["type"]
        assert 0 <= entity["hp"] <= entity["max_hp"]


def test_episode_strip_renders_a_grid_of_frames(snapshot):
    frames = []
    for tick in range(0, 60, 10):
        frame = dict(snapshot)
        frame["tick"] = tick
        frames.append(frame)

    fig = R.draw_episode_strip(frames, ncols=3)
    assert len(fig.axes) >= len(frames)
    plt.close(fig)
