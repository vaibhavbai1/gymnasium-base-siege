"""One legal-action random attack. No training or checkpoint required."""
import gymnasium as gym
import gymnasium_base_siege  # registers BaseSiege-v0


def main():
    env = gym.make("BaseSiege-v0", base_seed=42)
    env.action_space.seed(7)
    observation, info = env.reset(seed=0)
    while True:
        action = env.action_space.sample(mask=info["action_mask"])
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(f"Destruction: {100 * info['destruction']:.2f}%")
    print(f"Return: {info['episode_return']:.4f}; reason: {info['finish_reason']}")
    env.close()


if __name__ == "__main__":
    main()
