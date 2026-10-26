"""
Base Siege - board renderer.

Turns one recorded snapshot into a matplotlib drawing. Used three ways:

    Stage 5   eyeball the combat loop while debugging it
    Stage 6   look at a random episode that did something strange
    Stage 14  the Streamlit GUI imports this and adds widgets around it

The input is a plain dict, never live simulator objects. That is deliberate:
BUILD_ORDER_v2 requires GUI replay to work from recorded snapshots rather than
re-simulating, so the demo cannot diverge from the evidence in front of a
marker. It also means this module can be written and tested before combat.py
exists, and that the schema below IS the contract combat.record_snapshot()
must satisfy.

Two styles:

    style="debug"          flat coloured squares. Maximum legibility. Use this
                           while building. A square with a line coming out of
                           it is the clearest possible debugging signal.
    style="presentation"   original vector glyphs. Use this for the demo and
                           report figures.

Same data, same function, one branch. The two are not interchangeable: icons
make targeting bugs HARDER to spot, which is the opposite of what Stage 5
needs.

Backend note: this module does not call matplotlib.use(). A library that
forces a backend breaks its callers. Tests and headless scripts should set
matplotlib.use("Agg") themselves before importing pyplot.

--------------------------------------------------------------------------
SNAPSHOT SCHEMA
--------------------------------------------------------------------------
{
  "tick":                int,
  "phase":               "shopping" | "battle",
  "destroyed_fraction":  float,        # 0.0 .. 1.0
  "spent":               int,
  "troops_alive":        int,
  "troops_total":        int,
  "reserve":             {troop_type: int},          # bought, not deployed
  "buildings": [
      {"type": str, "row": int, "col": int, "hp": int, "max_hp": int}
  ],
  "troops": [
      {"type": str, "row": int, "col": int, "hp": int, "max_hp": int,
       "entity_id": int, "path": [[row, col], ...] | None}
  ],
  "attacks": [
      {"from": [row, col], "to": [row, col],
       "kind": "troop" | "defence", "damage": int}
  ]
}

Every value is JSON-serialisable, so a full episode can be written to disk and
replayed later without the simulator.
"""

from __future__ import annotations

from matplotlib.patches import (
    Arc,
    Circle,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

from gymnasium_base_siege.core import config as C

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Chosen to stay distinguishable in greyscale print, since the report will
# likely be marked on paper.

COLOURS = {
    "town_hall": "#6C63C8",
    "cannon": "#C74E77",
    "mortar": "#CE5A2E",
    "storage": "#7A8B8E",
    "wall": "#3B4652",
    "tank": "#1E8F6E",
    "melee": "#3179C6",
    "ranged": "#5E8F22",
    "breaker": "#D99425",
}

def _shade(hex_colour: str, factor: float) -> str:
    """Lighten (factor > 1) or darken (factor < 1) a hex colour."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(
        f"{min(255, max(0, int(v * factor))):02x}" for v in (r, g, b)
    )


GROUND = "#F3F1EC"
DEPLOY_ZONE = "#C9DCEA"
GRID_LINE = "#FFFFFF"
BOARD_EDGE = "#C7C4BC"

ATTACK_TROOP = "#14508C"
ATTACK_DEFENCE = "#A32D2D"
RANGE_RING = "#CE5A2E"
PATH_LINE = "#4A5568"

HP_BACKING = "#B4B2A9"
HP_GOOD = "#5E8F22"
HP_FAIR = "#D99425"
HP_POOR = "#D6423F"

HP_GOOD_ABOVE = 0.60
HP_FAIR_ABOVE = 0.25

# Glyph geometry is expressed in cell-fractions so it scales with the axes.
# Row 0 is at the top, so smaller y is "up".
BUILDING_SIZE = 0.92
TROOP_SIZE = 0.74


# ---------------------------------------------------------------------------
# Health bars
# ---------------------------------------------------------------------------

def _hp_colour(fraction: float) -> str:
    if fraction > HP_GOOD_ABOVE:
        return HP_GOOD
    if fraction > HP_FAIR_ABOVE:
        return HP_POOR if fraction <= 0 else HP_FAIR
    return HP_POOR


def _health_bar(row: int, col: int, fraction: float, is_troop: bool) -> list:
    """Two rectangles: grey backing, coloured foreground scaled by health."""
    fraction = max(0.0, min(1.0, fraction))
    width = TROOP_SIZE if is_troop else BUILDING_SIZE
    height = 0.08
    x = col - width / 2
    y = row - (0.46 if is_troop else 0.52)

    return [
        Rectangle((x, y), width, height, facecolor=HP_BACKING,
                  edgecolor="none", zorder=6),
        Rectangle((x, y), width * fraction, height,
                  facecolor=_hp_colour(fraction), edgecolor="none", zorder=7),
    ]


# ---------------------------------------------------------------------------
# Glyphs
# ---------------------------------------------------------------------------
# Each returns a list of patches drawn around cell centre (row, col). All are
# geometric primitives - rectangles, circles, one polygon or arc each. Nothing
# here derives from any existing game's artwork.

def _rounded(x, y, w, h, colour, radius=0.09, **kw):
    return FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=colour, edgecolor="none", **kw,
    )


def _glyph_town_hall(row, col, colour):
    return [
        _rounded(col - 0.34, row - 0.02, 0.68, 0.34, colour, zorder=4),
        Polygon([(col - 0.44, row - 0.02), (col, row - 0.36),
                 (col + 0.44, row - 0.02)],
                closed=True, facecolor=_shade(colour, 0.72),
                edgecolor="none", zorder=4),
        Rectangle((col - 0.03, row - 0.52), 0.06, 0.20,
                  facecolor=colour, edgecolor="none", zorder=4),
        Polygon([(col + 0.03, row - 0.52), (col + 0.28, row - 0.45),
                 (col + 0.03, row - 0.38)],
                closed=True, facecolor=colour, edgecolor="none", zorder=4),
    ]


def _glyph_cannon(row, col, colour):
    return [
        Rectangle((col - 0.10, row - 0.38), 0.20, 0.36,
                  facecolor=colour, edgecolor="none", zorder=4),
        Circle((col, row + 0.10), 0.28,
               facecolor=_shade(colour, 1.18), edgecolor="none", zorder=5),
    ]


def _glyph_mortar(row, col, colour):
    return [
        Polygon([(col - 0.26, row + 0.14), (col - 0.17, row - 0.32),
                 (col + 0.17, row - 0.32), (col + 0.26, row + 0.14)],
                closed=True, facecolor=colour, edgecolor="none", zorder=4),
        _rounded(col - 0.36, row + 0.10, 0.72, 0.26,
                 _shade(colour, 0.72), radius=0.08, zorder=5),
    ]


def _glyph_storage(row, col, colour):
    return [
        _rounded(col - 0.34, row - 0.30, 0.68, 0.62, colour,
                 radius=0.10, zorder=4),
        Rectangle((col - 0.34, row - 0.10), 0.68, 0.11,
                  facecolor=GROUND, alpha=0.45, edgecolor="none", zorder=5),
    ]


def _glyph_wall(row, col, colour):
    patches = [Rectangle((col - 0.5, row - 0.22), 1.0, 0.72,
                         facecolor=colour, edgecolor="none", zorder=4)]
    for offset in (-0.36, -0.06, 0.24):
        patches.append(
            Rectangle((col + offset, row - 0.48), 0.22, 0.28,
                      facecolor=colour, edgecolor="none", zorder=4)
        )
    return patches


def _glyph_tank(row, col, colour):
    return [
        _rounded(col - 0.34, row - 0.16, 0.68, 0.42, colour,
                 radius=0.12, zorder=8),
        Rectangle((col - 0.04, row - 0.04), 0.40, 0.09,
                  facecolor=_shade(colour, 0.65), edgecolor="none", zorder=9),
        Circle((col - 0.04, row + 0.01), 0.16,
               facecolor=_shade(colour, 0.65), edgecolor="none", zorder=10),
    ]


def _glyph_melee(row, col, colour):
    return [
        Circle((col - 0.06, row + 0.04), 0.22,
               facecolor=colour, edgecolor="none", zorder=8),
        Polygon([(col + 0.10, row + 0.16), (col + 0.34, row - 0.22),
                 (col + 0.24, row - 0.26), (col + 0.04, row + 0.10)],
                closed=True, facecolor=colour, edgecolor="none", zorder=9),
    ]


def _glyph_ranged(row, col, colour):
    return [
        Circle((col - 0.08, row + 0.04), 0.22,
               facecolor=colour, edgecolor="none", zorder=8),
        Arc((col + 0.14, row - 0.02), 0.40, 0.60, angle=0,
            theta1=-80, theta2=80, edgecolor=colour, linewidth=2.2, zorder=9),
    ]


def _glyph_breaker(row, col, colour):
    return [
        Circle((col - 0.08, row + 0.06), 0.22,
               facecolor=colour, edgecolor="none", zorder=8),
        Rectangle((col + 0.12, row - 0.26), 0.06, 0.42,
                  facecolor=colour, edgecolor="none", zorder=9),
        _rounded(col + 0.04, row - 0.34, 0.30, 0.16, colour,
                 radius=0.04, zorder=9),
    ]


GLYPHS = {
    "town_hall": _glyph_town_hall,
    "cannon": _glyph_cannon,
    "mortar": _glyph_mortar,
    "storage": _glyph_storage,
    "wall": _glyph_wall,
    "tank": _glyph_tank,
    "melee": _glyph_melee,
    "ranged": _glyph_ranged,
    "breaker": _glyph_breaker,
}


def _entity_patches(kind: str, row: int, col: int, style: str) -> list:
    """Flat square in debug style, glyph in presentation style."""
    colour = COLOURS[kind]

    if style == "presentation":
        return GLYPHS[kind](row, col, colour)

    size = BUILDING_SIZE if kind in C.BUILDING_TYPES else TROOP_SIZE
    zorder = 4 if kind in C.BUILDING_TYPES else 8
    return [
        _rounded(col - size / 2, row - size / 2, size, size, colour,
                 radius=0.10, zorder=zorder)
    ]


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

def _draw_board(ax) -> None:
    n = C.GRID_SIZE

    ax.add_patch(Rectangle((-0.5, -0.5), n, n, facecolor=GROUND,
                           edgecolor="none", zorder=0))

    for row, col in C.DEPLOY_TILES:
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1,
                               facecolor=DEPLOY_ZONE, edgecolor="none",
                               zorder=1))

    for i in range(n + 1):
        ax.plot([-0.5, n - 0.5], [i - 0.5, i - 0.5],
                color=GRID_LINE, linewidth=0.8, zorder=2)
        ax.plot([i - 0.5, i - 0.5], [-0.5, n - 0.5],
                color=GRID_LINE, linewidth=0.8, zorder=2)

    ax.add_patch(Rectangle((-0.5, -0.5), n, n, facecolor="none",
                           edgecolor=BOARD_EDGE, linewidth=1.0, zorder=12))

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(n - 0.3, -0.7)          # inverted: row 0 at the top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------

def _draw_ranges(ax, buildings) -> int:
    """
    A circle per defence at its attack radius, plus an inner circle at the
    mortar's minimum range.

    The inner circle is the point of the whole overlay: it is the mortar's
    blind spot, drawn rather than explained.
    """
    count = 0
    for b in buildings:
        stats = {**C.BUILDING_STATS[b["type"]],
                 **{k: b[k] for k in ("range", "min_range") if k in b}}
        if stats["range"] <= 0 or b["hp"] <= 0:
            continue

        ax.add_patch(Circle((b["col"], b["row"]), stats["range"],
                            facecolor="none",
                            edgecolor=RANGE_RING, linewidth=1.0, alpha=0.55,
                            linestyle=(0, (4, 4)), zorder=3))
        count += 1

        if stats["min_range"] > 0:
            ax.add_patch(Circle((b["col"], b["row"]), stats["min_range"],
                                facecolor=RANGE_RING, alpha=0.22,
                                edgecolor=RANGE_RING, linewidth=1.1,
                                linestyle="solid", zorder=3))
            count += 1
    return count


def _draw_paths(ax, troops) -> None:
    for t in troops:
        path = t.get("path")
        if not path or len(path) < 2:
            continue
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color=PATH_LINE, linewidth=1.1, alpha=0.55,
                linestyle=(0, (3, 3)), zorder=3)


def _draw_attacks(ax, attacks) -> None:
    for a in attacks:
        colour = ATTACK_TROOP if a["kind"] == "troop" else ATTACK_DEFENCE
        (r0, c0), (r1, c1) = a["from"], a["to"]
        ax.annotate(
            "", xy=(c1, r1), xytext=(c0, r0),
            arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.6,
                            shrinkA=6, shrinkB=8, alpha=0.95),
            zorder=11,
        )


def _draw_damage(ax, attacks) -> None:
    """
    Damage dealt this tick, annotated at the target tile.

    One mortar shot producing four numbers at once is how you confirm splash
    works without reading a log.
    """
    totals: dict[tuple[int, int], int] = {}
    for a in attacks:
        key = (a["to"][0], a["to"][1])
        totals[key] = totals.get(key, 0) + a.get("damage", 0)

    for (row, col), dmg in totals.items():
        if dmg <= 0:
            continue
        ax.text(col, row + 0.56, f"-{dmg}", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=ATTACK_DEFENCE,
                zorder=13,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.8))


def _draw_hud(ax, snapshot) -> None:
    destroyed = snapshot.get("destroyed_fraction", 0.0)
    parts = [
        f"tick {snapshot.get('tick', 0)}",
        f"destroyed {destroyed * 100:.0f}%",
        f"spent {snapshot.get('spent', 0)}/{C.RESOURCE_BUDGET}",
        f"troops {snapshot.get('troops_alive', 0)}/{snapshot.get('troops_total', 0)}",
        snapshot.get("phase", "battle"),
    ]
    ax.text(0.0, -0.035, "  ·  ".join(parts), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.5, color="#4A5568")

    reserve = snapshot.get("reserve") or {}
    if any(reserve.values()):
        text = "  ".join(f"{k[:3]} {v}" for k, v in reserve.items() if v)
        ax.text(1.0, -0.035, f"reserve: {text}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="#4A5568")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_snapshot(
    snapshot: dict,
    ax,
    style: str = "debug",
    show_ranges: bool = False,
    show_paths: bool = False,
    show_damage: bool = False,
    show_hud: bool = True,
    show_health: bool = True,
) -> None:
    """
    Draw one recorded snapshot onto an existing matplotlib Axes.

    Mutates `ax` and returns None, so the caller owns the figure and can put
    several snapshots side by side.
    """
    if style not in ("debug", "presentation"):
        raise ValueError(f"style must be 'debug' or 'presentation', got {style!r}")

    _draw_board(ax)

    if show_ranges:
        _draw_ranges(ax, snapshot.get("buildings", []))
    if show_paths:
        _draw_paths(ax, snapshot.get("troops", []))

    for b in snapshot.get("buildings", []):
        if b["hp"] <= 0:
            continue
        for patch in _entity_patches(b["type"], b["row"], b["col"], style):
            ax.add_patch(patch)
        if show_health and not (b["type"] == "wall" and style == "presentation"):
            for patch in _health_bar(b["row"], b["col"],
                                     b["hp"] / b["max_hp"], is_troop=False):
                ax.add_patch(patch)

    for t in snapshot.get("troops", []):
        if t["hp"] <= 0:
            continue
        for patch in _entity_patches(t["type"], t["row"], t["col"], style):
            ax.add_patch(patch)
        if show_health:
            for patch in _health_bar(t["row"], t["col"],
                                     t["hp"] / t["max_hp"], is_troop=True):
                ax.add_patch(patch)

    _draw_attacks(ax, snapshot.get("attacks", []))

    if show_damage:
        _draw_damage(ax, snapshot.get("attacks", []))
    if show_hud:
        _draw_hud(ax, snapshot)


def draw_episode_strip(snapshots, ncols: int = 5, every: int = 1, **kwargs):
    """
    Render a sequence of snapshots into one figure of small multiples.

    This is the Stage 5 "watch one episode" tool. Twenty minutes with this
    catches things no assertion will: troops stuck against walls, attack lines
    longer than the weapon's range, a tank strolling past a cannon.

    Returns the figure; the caller saves or shows it.
    """
    import matplotlib.pyplot as plt

    chosen = list(snapshots)[::every]
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows))
    flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    kwargs.setdefault("show_hud", False)
    kwargs.setdefault("show_health", False)

    for ax, snap in zip(flat, chosen):
        draw_snapshot(snap, ax, **kwargs)
        ax.set_title(f"tick {snap.get('tick', 0)}", fontsize=8, pad=4)

    for ax in flat[len(chosen):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def legend_handles(style: str = "presentation") -> list:
    """Patch handles for a matplotlib legend, in entity order."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=COLOURS[k], edgecolor="none",
              label=k.replace("_", " ").title())
        for k in ("town_hall", "cannon", "mortar", "storage", "wall",
                  "tank", "melee", "ranged", "breaker")
    ]


# ---------------------------------------------------------------------------
# Development fixture
# ---------------------------------------------------------------------------

def example_snapshot() -> dict:
    """
    A hand-built mid-battle snapshot.

    Exists so this module can be exercised before combat.py is written, and so
    tests have a fixture that does not depend on the simulator. Delete or keep
    once Stage 5 lands - a real record_snapshot() output should be a drop-in
    replacement, and if it is not, the schema drifted.
    """
    r0, c0, r1, c1 = C.CENTRED_RING_BOX

    walls = []
    for c in range(c0, c1 + 1):
        walls.append((r0, c))
        if c != 7:                                  # breached at column 7
            walls.append((r1, c))
    for r in range(r0 + 1, r1):
        walls.append((r, c0))
        walls.append((r, c1))

    buildings = [
        {"type": "wall", "row": r, "col": c, "hp": 300, "max_hp": 300}
        for r, c in walls
    ] + [
        {"type": "town_hall", "row": 7, "col": 7, "hp": 1500, "max_hp": 1500},
        {"type": "cannon", "row": 5, "col": 5, "hp": 600, "max_hp": 600},
        {"type": "cannon", "row": 5, "col": 10, "hp": 600, "max_hp": 600},
        {"type": "cannon", "row": 10, "col": 6, "hp": 270, "max_hp": 600},
        {"type": "mortar", "row": 8, "col": 6, "hp": 150, "max_hp": 500},
        {"type": "mortar", "row": 7, "col": 10, "hp": 500, "max_hp": 500},
        {"type": "storage", "row": 4, "col": 4, "hp": 800, "max_hp": 800},
        {"type": "storage", "row": 4, "col": 11, "hp": 800, "max_hp": 800},
        {"type": "storage", "row": 11, "col": 4, "hp": 800, "max_hp": 800},
        {"type": "storage", "row": 11, "col": 9, "hp": 160, "max_hp": 800},
    ]

    troops = [
        {"type": "tank", "row": 9, "col": 6, "hp": 630, "max_hp": 900,
         "entity_id": 1, "path": [[9, 6], [8, 6]]},
        {"type": "melee", "row": 11, "col": 8, "hp": 48, "max_hp": 120,
         "entity_id": 2, "path": [[11, 8], [11, 9]]},
        {"type": "melee", "row": 13, "col": 8, "hp": 120, "max_hp": 120,
         "entity_id": 3, "path": [[13, 8], [12, 8], [11, 8]]},
        {"type": "ranged", "row": 13, "col": 7, "hp": 76, "max_hp": 90,
         "entity_id": 4, "path": None},
    ]

    attacks = [
        {"from": [9, 6], "to": [8, 6], "kind": "troop", "damage": 8},
        {"from": [11, 8], "to": [11, 9], "kind": "troop", "damage": 15},
        {"from": [13, 7], "to": [10, 6], "kind": "troop", "damage": 12},
        {"from": [10, 6], "to": [11, 8], "kind": "defence", "damage": 5},
        {"from": [7, 10], "to": [13, 7], "kind": "defence", "damage": 9},
    ]

    return {
        "tick": 140,
        "phase": "battle",
        "destroyed_fraction": 0.34,
        "spent": 199,
        "troops_alive": 4,
        "troops_total": 16,
        "reserve": {"tank": 0, "melee": 2, "ranged": 1, "breaker": 0},
        "buildings": buildings,
        "troops": troops,
        "attacks": attacks,
    }
