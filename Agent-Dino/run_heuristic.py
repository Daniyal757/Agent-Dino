from dino_env import DinoEnv
import numpy as np

env = DinoEnv()
obs, _ = env.reset()

while True:
    # Use last frame from obs (channel-last)
    frame = obs[..., -1]

    # Look at a band where cactus usually appears (right-middle)
    band = frame[40:70, 50:80]
    # Count dark pixels (cactus)
    cactus_score = np.mean(band < 150)

    if cactus_score > 0.15:
        action = 1  # jump
    else:
        action = 0  # do nothing

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated:
        obs, _ = env.reset()
