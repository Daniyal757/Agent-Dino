dino_env.py
This is the game environment your agent interacts with.
It launches Chrome Dino using Playwright, grabs screenshots, applies actions (jump / nothing), detects crashes, and returns observations + rewards.

Think of it as: “The world your agent lives in.”

train.py
This file trains your RL agent using PPO (or whatever algorithm you choose).
It loads the environment, runs episodes, updates the model, and saves checkpoints.

Think of it as: “Teach the agent how to play.”

run_agent.py
This loads your trained model and lets it play the game in real time.
No training — just watching the agent perform.

Think of it as: “Show me what the agent learned.”

run_heuristic.py
A simple rule‑based player (no machine learning).
Usually: jump when a cactus is close.

Think of it as: “A basic bot to compare against the RL agent.”

debug_env.py
A quick script to test the environment without training.
Shows the cropped frames, checks if Dino starts correctly, and ensures screenshots work.

Think of it as: “Make sure the environment isn’t broken.”

requirements.txt
List of Python packages your project needs (gymnasium, stable‑baselines3, playwright, numpy, etc.).

Think of it as: “Install everything needed for the project.”

.gitignore
Tells Git which files NOT to upload (cache, logs, models, browser binaries, etc.).

Think of it as: “Keep the repo clean.”

.vscode/settings.json
VS Code editor settings — formatting, linting, Python interpreter, etc.

Think of it as: “Editor preferences for this project.”
