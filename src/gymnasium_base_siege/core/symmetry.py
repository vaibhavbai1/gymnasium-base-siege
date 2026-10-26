"""D4 coordinate views; the underlying combat trajectory is not rotated."""
import gymnasium as gym
import numpy as np
from gymnasium_base_siege.core import config as C


def transform_tile(tile, transform):
    if transform not in range(8):
        raise ValueError("D4 transform must be in 0..7")
    r, c = tile
    if transform >= 4:
        c = C.GRID_SIZE - 1 - c
    for _ in range(transform % 4):
        r, c = c, C.GRID_SIZE - 1 - r
    return r, c


def action_permutation(transform):
    permutation = np.arange(C.N_ACTIONS)
    lookup = {tile: i for i, tile in enumerate(C.DEPLOY_TILES)}
    for kind in range(len(C.TROOP_TYPES)):
        for index, tile in enumerate(C.DEPLOY_TILES):
            start = C.ACTION_DEPLOY_FIRST + kind * C.N_DEPLOY_TILES
            permutation[start + index] = start + lookup[transform_tile(tile, transform)]
    return permutation


def transform_observation(obs, transform):
    grid = obs["grid"]
    if transform >= 4:
        grid = np.flip(grid, axis=-1)
    grid = np.rot90(grid, k=-(transform % 4), axes=(-2, -1))
    return {"grid": np.ascontiguousarray(grid), "vector": obs["vector"].copy()}


class SymmetryWrapper(gym.Wrapper):
    def __init__(self, env, transform=None):
        super().__init__(env)
        if transform is not None and transform not in range(8):
            raise ValueError("D4 transform must be in 0..7")
        self.fixed_transform = transform
        self.transform = 0
        self.permutation = action_permutation(0)
        self.inverse = np.argsort(self.permutation)
        self.view_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.view_rng = np.random.default_rng(np.random.SeedSequence([seed, 404]))
        # Sampling a view must not change the sequence of training layouts.
        self.transform = self.fixed_transform if self.fixed_transform is not None else int(self.view_rng.integers(8))
        self.permutation = action_permutation(self.transform)
        self.inverse = np.argsort(self.permutation)
        return transform_observation(obs, self.transform), {**info, "view": self.transform}

    def action_masks(self):
        return self.env.action_masks()[self.inverse]

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Unknown action")
        obs, reward, terminated, truncated, info = self.env.step(int(self.inverse[int(action)]))
        return transform_observation(obs, self.transform), reward, terminated, truncated, {**info, "view": self.transform}

    def set_fixed_army(self, value):
        self.env.set_fixed_army(value)
