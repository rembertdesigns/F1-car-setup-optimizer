import gymnasium as gym
from gymnasium import spaces
import numpy as np

class F1SetupEnv(gym.Env):
    """
    Custom F1 Car Setup Environment for RL Training
    """
    def __init__(self):
        super(F1SetupEnv, self).__init__()

        # Setup parameters range
        self.low = np.array([0, 0, 30, 1, 50], dtype=np.float32)
        self.high = np.array([50, 50, 50, 11, 60], dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        self.state = self.reset()[0]

    def step(self, action):
        # Apply action as relative change
        action = np.clip(action, -1, 1)
        delta = (self.high - self.low) * 0.05 * action  # 5% adjustment
        self.state = np.clip(self.state + delta, self.low, self.high)

        # Define a dummy reward function (example: minimize ride height + maximize aero balance)
        front_wing, rear_wing, ride_height, suspension, brake_bias = self.state
        downforce = front_wing + rear_wing
        reward = -ride_height + 0.1 * downforce - 0.5 * abs(brake_bias - 54)

        terminated = False
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(self.low, self.high).astype(np.float32)
        return self.state, {}

    def render(self):
        print(f"Current Setup: {self.state}")
