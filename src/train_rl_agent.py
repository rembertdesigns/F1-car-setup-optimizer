import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.setup_env import F1SetupEnv  # ✅ correct class name now

# Create environment instance
env = F1SetupEnv()
check_env(env, warn=True)  # ✅ Verifies your env is compliant with Gymnasium

# Define model save directory
save_path = "models/rl"
os.makedirs(save_path, exist_ok=True)

# Define checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=save_path,
    name_prefix="ppo_car_setup_agent"
)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/rl/",
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
    learning_rate=3e-4
)

# Train the model
model.learn(
    total_timesteps=100_000,
    callback=checkpoint_callback
)

# Save the final model
model.save(os.path.join(save_path, "ppo_car_setup_final"))
print("✅ PPO agent training complete and model saved.")
