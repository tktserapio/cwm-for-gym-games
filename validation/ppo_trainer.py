import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

class PPOTrainer:
    def __init__(self, env_class):
        self.env_class = env_class

    def train_and_evaluate(self, total_timesteps=100_000):
        """
        Trains a PPO agent via self-play logic.
        """
        # Vectorize environment
        vec_env = DummyVecEnv([lambda: self.env_class()])

        # Initialize PPO with hyperparameters from gg-bench Table 4 [cite: 2748]
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            batch_size=64,
            n_steps=2048,
            verbose=1
        )

        try:
            # Train the agent
            model.learn(total_timesteps=total_timesteps)
            
            # Evaluate against Random Agent
            win_rate = self._evaluate_winrate(model)
            return win_rate
            
        except Exception as e:
            print(f"Training failed: {e}")
            return 0.0

    def _evaluate_winrate(self, model, episodes=20):
        """
        Evaluates the trained agent against a random opponent.
        """
        env = self.env_class()
        wins = 0
        
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            # Simple evaluation: Agent plays P1, Random plays P2
            while not done:
                if env.unwrapped.current_player == 1: # Assuming 1 is Agent
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample() # Random opponent
                    
                # Ensure action is valid (masking)
                if action not in env.valid_moves():
                    action = np.random.choice(env.valid_moves())

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if done and reward == 1 and env.unwrapped.current_player == 1:
                    wins += 1
                    
        return wins / episodes