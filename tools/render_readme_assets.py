"""Render real simulator states for the README; no trained policy is used.

From the repository root with Python 3.12:
    python -m pip install -e ".[render]"
    python tools/render_readme_assets.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gymnasium as gym
import gymnasium_base_siege as siege
from gymnasium_base_siege.core.render import draw_snapshot, legend_handles

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#233247"})
MOTIFS = ("closed_ring", "ns_gates", "ew_gates", "layered_core")
LABELS = ("Closed ring", "North–south gates", "East–west gates", "Layered core")


def overview():
    fig, axes = plt.subplots(2, 2, figsize=(10, 11), dpi=150)
    fig.suptitle("BASE SIEGE", fontsize=23, weight="bold", y=.975)
    fig.text(.5, .938, "Four procedural layout types · 16 × 16 tiles", ha="center", fontsize=12)
    for ax, motif, label in zip(axes.flat, MOTIFS, LABELS):
        env = gym.make("BaseSiege-v0", base_seed=42, motifs=(motif,))
        env.reset(seed=0)
        draw_snapshot(env.unwrapped.snapshot(), ax, style="presentation", show_hud=False)
        ax.set_title(label, fontsize=13, weight="bold", pad=8)
        env.close()
    fig.legend(handles=legend_handles()[:5], loc="lower center", ncol=5,
               bbox_to_anchor=(.5, .040), frameon=False, fontsize=11)
    fig.text(.5, .025, "Blue border tiles: deployment zone  |  Green bars: remaining health",
             ha="center", fontsize=10)
    fig.text(.5, .008, "Actual simulator renderings · Default settings · Layout seed 42",
             ha="center", fontsize=9, color="#64748B")
    fig.subplots_adjust(left=.035, right=.965, top=.89, bottom=.095, hspace=.12, wspace=.10)
    fig.savefig(OUT / "base-siege-layouts.png", facecolor="white")
    plt.close(fig)


def attack():
    env = gym.make("BaseSiege-v0", base_seed=42, motifs=("ns_gates",),
                   fixed_army=True, record_ticks=True)
    _, info = env.reset(seed=0)
    # Fixed, openly documented demonstration: deploy near the north gate,
    # then wait. This is not a learned policy or an evaluation experiment.
    order = ["tank", "ranged", "melee", "breaker", "ranged", "tank",
             "melee", "ranged", "breaker", "melee", "tank", "ranged",
             "melee", "breaker", "melee", "melee"]
    actions = [siege.deployment_action(kind, (7, 8, 6)[i % 3])
               for i, kind in enumerate(order)]
    finished = False
    i = 0
    while not finished:
        action = actions[i] if i < len(actions) else 245
        assert info["action_mask"][action], f"Illegal scripted action: {action}"
        _, _, terminated, truncated, info = env.step(action)
        finished = terminated or truncated
        i += 1
    # One real snapshot per tick, including the final state.
    ticks = {frame["tick"]: frame for frame in env.unwrapped.visual_history}
    selected = [ticks[t] for t in sorted(ticks) if t % 4 == 0 or t == max(ticks)]
    frames = []
    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=100)
    fig.subplots_adjust(left=.04, right=.96, top=.83, bottom=.12)
    fig.text(.5, .97, "BASE SIEGE · AN ATTACK IN MOTION", ha="center", fontsize=16, weight="bold")
    fig.text(.5, .935, "Scripted deployment · Fixed army · Default settings", ha="center", fontsize=11)
    status = fig.text(.5, .885, "", ha="center", fontsize=12)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               bbox_to_anchor=(.5, .035), frameon=False, fontsize=9)
    fig.text(.5, .012, "Illustration only; no trained agent. Animation playback is accelerated.",
             ha="center", fontsize=9, color="#64748B")
    for snap in selected:
        ax.clear()
        draw_snapshot(snap, ax, style="presentation", show_hud=False)
        status.set_text(f"Tick {snap['tick']:03d} / 300   |   Live troops: {snap['troops_alive']}"
                        f"   |   Reserve: {sum(snap['reserve'].values())}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    # Choose a mid-episode state for the static fallback, not a fabricated scene.
    mid = min(range(len(selected)), key=lambda i: abs(selected[i]["tick"] - 120))
    frames[mid].save(OUT / "base-siege-attack.png")
    durations = [100] * len(frames)
    durations[0] = 800
    durations[-1] = 1800
    frames[0].save(OUT / "base-siege-attack.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    plt.close(fig)
    metadata = {"package_version": siege.__version__, "layout_seed": 42,
                "reset_seed": 0, "motif": "ns_gates", "fixed_army": True,
                "policy": "scripted deployment, then wait; no trained model",
                "deployment_actions": actions, "wait_action": 245,
                "static_image_tick": selected[mid]["tick"],
                "animation_ticks": [s["tick"] for s in selected],
                "finish_reason": info["finish_reason"]}
    (OUT / "render-details.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env.close()
    print(f"Created overview, attack PNG and {len(frames)}-frame GIF in {OUT}")


if __name__ == "__main__":
    overview()
    attack()
