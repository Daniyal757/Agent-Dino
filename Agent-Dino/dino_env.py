import gymnasium as gym
import numpy as np
import cv2
import time
from playwright.sync_api import sync_playwright

class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def render(self):
        frame = self.frames[-1]
        cv2.imshow("Dino RL", frame)
        cv2.waitKey(1)

    def __init__(self):
        super().__init__()

        # Actions: 0 = nothing, 1 = jump, 2 = duck
        self.action_space = gym.spaces.Discrete(3)

        # Observation: 84x84 grayscale stacked frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 4), dtype=np.uint8
        )

        # Launch browser
        self.p = sync_playwright().start()
        self.browser = self.p.chromium.launch(headless=False)
        self.page = self.browser.new_page()

        # Load stable Dino version (NO ads, NO overlays)
        self.page.goto("https://elgoog.im/t-rex/")

        # Wait for game to load
        time.sleep(1.5)

        # Start game
        self.page.keyboard.press("Space")
        time.sleep(1.0)

        # Frame stack
        self.frames = []

    def get_frame(self):
        # Correct region for elgoog Dino
        img_bytes = self.page.screenshot(clip={
            "x": 0,
            "y": 150,
            "width": 600,
            "height": 200
        })

        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(frame, (84, 84))

        return frame

    def step(self, action):
        # Apply action
        if action == 1:
            self.page.keyboard.press("Space")
        elif action == 2:
            self.page.keyboard.down("ArrowDown")
        else:
            self.page.keyboard.up("ArrowDown")

        frame = self.get_frame()

        # Update frame stack
        self.frames.append(frame)
        if len(self.frames) > 4:
            self.frames.pop(0)

        obs = np.stack(self.frames, axis=-1)

        # Crash detection for elgoog Dino
        dead = self.page.evaluate("Runner.instance_.crashed")

        reward = 1
        terminated = False

        if dead:
            reward = -100
            terminated = True

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Restart game
        self.page.reload()
        time.sleep(1.5)
        self.page.keyboard.press("Space")
        time.sleep(1.0)

        # Reset frame stack
        self.frames = []
        frame = self.get_frame()

        for _ in range(4):
            self.frames.append(frame)

        obs = np.stack(self.frames, axis=-1)
        return obs, {}

    def close(self):
        self.browser.close()
        self.p.stop()
