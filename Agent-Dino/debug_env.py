from dino_env import DinoEnv

env = DinoEnv()
obs, _ = env.reset()

for i in range(50):
    obs, reward, terminated, truncated, info = env.step(0)
    env.render()
    if terminated:
        print("Terminated at step", i, "reward:", reward)
        obs, _ = env.reset()
