import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym
import numpy as np
import pytest
import gymnasium_base_siege
from gymnasium.utils.env_checker import check_env
from gymnasium_base_siege import BaseSiegeEnv, deployment_action
from gymnasium_base_siege.core import config as C
from gymnasium_base_siege.core.combat import BattleState, step_tick, deploy_troop
from gymnasium_base_siege.core.entities import Building, Troop
from gymnasium_base_siege.core.pathfinding import GridState
from gymnasium_base_siege.settings import resolve_settings


def test_gymnasium_checker_with_masked_test_sampler(monkeypatch):
    # check_env otherwise samples actions that this task intentionally forbids.
    # Only this test supplies a legal-action sampler; the public space is unchanged.
    env = BaseSiegeEnv()
    sample = env.action_space.sample
    def masked_sample(mask=None, probability=None):
        if mask is None and probability is None:
            mask = env.action_masks().astype(np.int8)
        return sample(mask=mask, probability=probability)
    monkeypatch.setattr(env.action_space, "sample", masked_sample)
    check_env(env, skip_render_check=True)


def play(seed=3, **kwargs):
    env = gym.make("BaseSiege-v0", **kwargs)
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)
    records = []
    while True:
        assert env.observation_space.contains(obs)
        assert info["action_mask"].dtype == np.int8
        action = env.action_space.sample(mask=info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        records.append((action, reward, info["tick"], info["destruction"]))
        if terminated or truncated:
            assert not info["action_mask"].any()
            break
    env.close()
    return records


@pytest.mark.parametrize("motif", C.MOTIFS)
def test_registered_complete_episode_repeatability(motif):
    assert play(motifs=(motif,)) == play(motifs=(motif,))


def test_import_on_make():
    env = gym.make("gymnasium_base_siege:BaseSiege-v0")
    env.reset(seed=0)
    assert env.action_space.n == 246
    assert env.spec.id == "BaseSiege-v0"
    assert env.unwrapped.spec.id == "BaseSiege-v0"
    assert env.unwrapped.base_spec.motif in C.MOTIFS
    env.close()


def test_illegal_action_preserves_state():
    env = BaseSiegeEnv(record=True)
    env.reset(seed=0)
    before = json.dumps(env.snapshot(), sort_keys=True)
    with pytest.raises(ValueError, match="Illegal"):
        env.step(245)
    assert json.dumps(env.snapshot(), sort_keys=True) == before


def test_empty_army_termination_and_cost_once():
    env = BaseSiegeEnv()
    env.reset(seed=0)
    _, reward, done, truncated, info = env.step(4)
    assert done and not truncated and reward == 0 and info["tick"] == 0
    with pytest.raises(gym.error.ResetNeeded):
        env.step(4)
    env.reset(seed=0)
    for _ in range(25):
        env.step(1)
    assert np.flatnonzero(env.action_masks()).tolist() == [4]
    assert env.step(4)[1] == -0.3
    assert env.step(245)[1] == 0


def test_timing_and_termination_at_battle_limit():
    env = BaseSiegeEnv()
    env.reset(seed=0)
    env.step(0)
    assert env.state.tick == 0
    env.step(4)
    for _ in range(29):
        assert env.step(245)[2:4] == (False, False)
    _, _, done, truncated, info = env.step(245)
    assert done and not truncated and info["tick"] == 300


def test_safety_guard_truncation():
    env = BaseSiegeEnv()
    env.reset(seed=0)
    env.decisions = 69  # force otherwise unreachable safety boundary
    assert env.step(0)[2:4] == (False, True)


def test_settings_isolation_cost_masks_and_roundtrip():
    supplied = {"tank": {"hp": 1500, "speed": 0.20, "damage_per_tick": 20, "cost": 80}}
    custom = BaseSiegeEnv(troop_stats=supplied, defence_stats={"cannon": {"hp": 1200, "range": 6}})
    standard = BaseSiegeEnv()
    supplied["tank"]["hp"] = 1
    for env in (custom, standard):
        env.reset(seed=2)
        env.step(0)
        env.step(0)
    assert not custom.action_masks()[0] and standard.action_masks()[0]
    assert custom.state.spent == 160 and standard.state.spent == 50
    for env in (custom, standard):
        env.step(4)
        deploy_troop(env.state, "tank", (0, 0))
    assert custom.state.troops[0].max_hp == 1500
    assert standard.state.troops[0].max_hp == 900
    assert custom.state.troops[0].speed == 0.20
    assert custom.state.grid.entry_cost("tank", (0, 1)) == 5
    assert standard.state.grid.entry_cost("tank", (0, 1)) == 10
    assert next(b for b in custom.state.buildings if b.building_type == "cannon").max_hp == 1200
    saved = json.loads(json.dumps(custom.settings))
    restored = BaseSiegeEnv(**saved)
    assert restored.settings == custom.settings
    assert restored.settings_hash == custom.settings_hash
    exported = custom.settings
    exported["troop_stats"]["tank"]["hp"] = 1
    assert custom.settings["troop_stats"]["tank"]["hp"] == 1500
    assert C.TROOP_STATS["tank"]["hp"] == 900
    assert C.BUILDING_STATS["cannon"]["range"] == 3
    # No module-global state is changed: interleaved and threaded episodes match.
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(play, troop_stats={"melee": {"speed": 0.50}})
        b = pool.submit(play)
        assert a.result() == play(troop_stats={"melee": {"speed": 0.50}})
        assert b.result() == play()


def test_actual_damage_and_cooldown_follow_settings():
    troops, buildings = resolve_settings({"melee": {"damage_per_tick": 30}},
                                        {"cannon": {"damage_per_shot": 11, "attack_cooldown": 3}})
    cannon = Building.spawn("cannon", (2, 2), stats=buildings["cannon"])
    troop = Troop.spawn("melee", (2, 1), 1, stats=troops["melee"])
    state = BattleState(GridState({(2, 2): cannon}, troop_stats=troops), troops=[troop])
    step_tick(state)
    assert cannon.hp == 570 and troop.hp == 109
    step_tick(state)
    assert cannon.hp == 540 and troop.hp == 109
    step_tick(state)
    assert troop.hp == 109
    step_tick(state)
    assert troop.hp == 98


def test_custom_movement_rate():
    troops, _ = resolve_settings({"tank": {"speed": 0.25}})
    troop = Troop.spawn("tank", (0, 0), 1, stats=troops["tank"])
    assert [troop.accumulate_movement() for _ in range(4)] == [False, False, False, True]


def test_zero_damage_defence_has_no_splash_damage():
    _, buildings = resolve_settings(defence_stats={"mortar": {"damage_per_shot": 0}})
    mortar = Building.spawn("mortar", (5, 5), stats=buildings["mortar"])
    troops = [Troop.spawn("tank", tile, i) for i, tile in enumerate(((5, 7), (5, 8)))]
    state = BattleState(GridState({(5, 5): mortar}), troops=troops)
    step_tick(state)
    assert [t.hp for t in troops] == [900, 900]


def test_custom_path_cost_and_observation_range():
    custom = BaseSiegeEnv(troop_stats={"melee": {"speed": 0.25, "damage_per_tick": 50}},
                          defence_stats={"cannon": {"range": 20}})
    original = BaseSiegeEnv()
    obs, _ = custom.reset(seed=0)
    normal, _ = original.reset(seed=0)
    wall = next(b for b in custom.state.buildings if b.is_wall)
    assert custom.state.grid.entry_cost("melee", wall.position) == 4 + 300 / 50
    assert obs["grid"][6].sum() > normal["grid"][6].sum()
    assert custom.observation_space.contains(obs)


@pytest.mark.parametrize("kwargs", [
    {"troop_stats": {"tank": {"hp": 0}}},
    {"troop_stats": {"tank": {"speed": 0}}},
    {"troop_stats": {"tank": {"speed": 1.1}}},
    {"troop_stats": {"tank": {"speed": 0.125}}},
    {"troop_stats": {"tank": {"hp": True}}},
    {"troop_stats": {"tank": {"damage_per_tick": float("nan")}}},
    {"troop_stats": {"tank": {"range": 0}}},
    {"troop_stats": {"dragon": {"hp": 100}}},
    {"troop_stats": {"tank": {"damage": 10}}},
    {"defence_stats": {"cannon": {"attack_cooldown": 0}}},
    {"defence_stats": {"mortar": {"min_range": 6}}},
    {"defence_stats": {"mortar": {"damage_per_shot": 1.5}}},
    {"defence_stats": {"wall": {"hp": 600}}},
    {"fixed_army": True, "troop_stats": {"tank": {"cost": 100}}},
])
def test_invalid_settings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        BaseSiegeEnv(**kwargs)


def test_render_and_pickle():
    env = BaseSiegeEnv(render_mode="rgb_array", defence_stats={"cannon": {"range": 6}})
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8
    assert next(b for b in env.snapshot()["buildings"] if b["type"] == "cannon")["range"] == 6
    restored = pickle.loads(pickle.dumps(env))
    assert restored.settings == env.settings
    assert restored.snapshot() == env.snapshot()
    assert BaseSiegeEnv().render() is None


def test_sync_vector_environment_masks():
    env = gym.vector.SyncVectorEnv([
        lambda: gym.make("BaseSiege-v0"),
        lambda: gym.make("BaseSiege-v0", troop_stats={"tank": {"speed": 0.20}}),
    ])
    obs, info = env.reset(seed=0)
    assert info["action_mask"].shape == (2, 246)
    assert info["settings_hash"][0] != info["settings_hash"][1]
    obs, *_ = env.step(np.array([0, 0]))
    assert obs["grid"].shape == (2, 8, 16, 16)
    env.close()
