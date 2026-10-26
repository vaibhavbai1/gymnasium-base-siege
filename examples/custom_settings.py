"""Two independent environments, with default and customized rules."""
import json
import gymnasium as gym
import gymnasium_base_siege


def main():
    standard = gym.make("BaseSiege-v0")
    custom = gym.make("BaseSiege-v0",
        troop_stats={"tank": {"hp": 1200, "speed": 0.20, "damage_per_tick": 12}},
        defence_stats={"cannon": {"hp": 800, "damage_per_shot": 8}})
    for label, env in (("Default", standard), ("Custom", custom)):
        _, info = env.reset(seed=42)
        print(label, json.dumps(env.unwrapped.settings["troop_stats"]["tank"]))
        print("Settings hash:", info["settings_hash"])
        env.close()


if __name__ == "__main__":
    main()
