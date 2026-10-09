from stable_baselines3 import PPO
from dino_env import DinoEnv

model = PPO.load("dino_pixel_agent")
env = DinoEnv()

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    # WATCH IT IN REAL TIME
    env.render()

    if done:
        obs, _ = env.reset()
