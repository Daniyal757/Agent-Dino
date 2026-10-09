from stable_baselines3 import PPO
from dino_env import DinoEnv

env = DinoEnv()

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

model.learn(total_timesteps=3_000_000)
model.save("dino_pixel_agent")

env.close()

