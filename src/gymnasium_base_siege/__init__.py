"""Import this package to register BaseSiege-v0 with Gymnasium."""
from gymnasium.envs.registration import register

from .environment import BaseSiegeEnv
from .core.generator import BaseSpec, generate_base
from .core.siege_env import deployment_action, describe_action

__version__ = "0.1.0rc1"
__all__ = ["BaseSiegeEnv", "BaseSpec", "generate_base", "deployment_action", "describe_action"]

# The simulator already owns its 300-tick terminal rule and 70-decision guard.
# Adding a TimeLimit here would change the termination contract.
register(id="BaseSiege-v0", entry_point="gymnasium_base_siege:BaseSiegeEnv")
