import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ==========================================
# 1. Wrapper for Board Games (Chess, TTT)
# ==========================================
class PerfectInfoWrapper(gym.Wrapper):
    """
    For symmetric or directional board games (Perfect Information).
    - Flips values (-1 -> 1) so Agent always sees itself as Player 1.
    - Rotates board 180 degrees if board_shape is provided (for directional games).
    """
    def __init__(self, env, board_shape=None):
        super().__init__(env)
        self.agent_player = None
        self.board_shape = board_shape

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.agent_player = np.random.choice([1, -1])
        info['agent_player'] = self.agent_player
        
        # If Agent is Player 2, Opponent (P1) moves first
        if self.agent_player == -1:
            valid_moves = self.env.unwrapped.valid_moves()
            if valid_moves:
                action = np.random.choice(valid_moves)
                obs, _, _, _, info = self.env.step(action)
        
        return self.canonicalize(obs, self.agent_player), info

    def step(self, action):
        # Transform action (if board is rotated)
        real_action = self.transform_action(action, self.agent_player)
        
        # Agent Move
        obs, reward, terminated, truncated, info = self.env.step(real_action)

        # Terminate on invalid move penalty
        if reward == -10:
            terminated = True
            return self.canonicalize(obs, self.agent_player), reward, terminated, truncated, info

        # Opponent Move
        if not (terminated or truncated):
            if self.env.unwrapped.current_player != self.agent_player:
                valid_moves = self.env.unwrapped.valid_moves()
                if valid_moves:
                    opp_action = np.random.choice(valid_moves)
                    obs, opp_reward, terminated, truncated, info = self.env.step(opp_action)
                    
                    if terminated or truncated:
                        reward = -opp_reward
                    else:
                        reward = 0

        return self.canonicalize(obs, self.agent_player), reward, terminated, truncated, info

    def canonicalize(self, obs, player):
        if player == 1: return obs
        obs = obs * -1 # Value flip
        if self.board_shape: # Spatial flip
            grid = obs.reshape(self.board_shape)
            grid = np.rot90(grid, 2)
            obs = grid.flatten()
        return obs

    def transform_action(self, action, player):
        if player == 1 or self.board_shape is None: return action
        # Invert index for 180 rotation
        total_squares = self.board_shape[0] * self.board_shape[1]
        return (total_squares - 1) - action

# ==========================================
# 2. Wrapper for Card Games (Poker)
# ==========================================
class ImperfectInfoWrapper(gym.Wrapper):
    """
    For Card Games (Imperfect Information).
    - Does NOT flip observations (relies on Env to give relative view).
    - Handles Opponent turns and Reward flipping.
    """
    def __init__(self, env):
        super().__init__(env)
        self.agent_player = None

    def reset(self, **kwargs):
        self.agent_player = np.random.choice([1, -1])
        obs, info = self.env.reset(**kwargs)
        info['agent_player'] = self.agent_player
        
        # If Agent is Player 2, Opponent (P1) moves first
        if self.agent_player == -1:
             if self.env.unwrapped.current_player != self.agent_player:
                valid_moves = self.env.unwrapped.valid_moves()
                if valid_moves:
                    action = np.random.choice(valid_moves)
                    obs, _, _, _, info = self.env.step(action)
        return obs, info

    def step(self, action):
        # Agent Move
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if reward == -10:
            terminated = True
            return obs, reward, terminated, truncated, info

        # Opponent Move
        if not (terminated or truncated):
            if self.env.unwrapped.current_player != self.agent_player:
                valid_moves = self.env.unwrapped.valid_moves()
                if valid_moves:
                    opp_action = np.random.choice(valid_moves)
                    obs, opp_reward, terminated, truncated, info = self.env.step(opp_action)
                    
                    if terminated or truncated:
                        reward = -opp_reward
                    else:
                        reward = 0
        return obs, reward, terminated, truncated, info

# ==========================================
# 3. Main Trainer Class
# ==========================================
class PPOTrainer:
    def __init__(self, env_class, game_type="perfect", board_shape=None):
        """
        env_class: The Gym class generated by LLM.
        game_type: "perfect" (Board) or "imperfect" (Cards).
        board_shape: Tuple (e.g. (8,8)) for directional board games, None otherwise.
        """
        self.env_class = env_class
        self.game_type = game_type
        self.board_shape = board_shape

    def _make_wrapped_env(self):
        """Factory function to create the correct environment."""
        env = self.env_class()
        if self.game_type == "perfect":
            return PerfectInfoWrapper(env, self.board_shape)
        else:
            return ImperfectInfoWrapper(env)

    def train_and_evaluate(self, total_timesteps=100_000):
        # Vectorize using the factory method
        vec_env = DummyVecEnv([self._make_wrapped_env])

        model = PPO(
            "MlpPolicy",
            vec_env,
            ent_coef=0.01, # Helps exploration
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            batch_size=64,
            n_steps=2048,
            verbose=0
        )

        try:
            print(f"Training ({self.game_type}) for {total_timesteps} steps...")
            model.learn(total_timesteps=total_timesteps)
            return self._evaluate_winrate(model)
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _evaluate_winrate(self, model, episodes=100):
        """
        Evaluates using the SAME wrapper logic used in training.
        """
        # We must use the wrapper during eval to ensure Agent sees the right perspective
        eval_env = self._make_wrapped_env()
        
        wins = 0
        losses = 0
        draws = 0
        
        for ep in range(episodes):
            obs, info = eval_env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                # Check for validity (optional safety check)
                valid_moves = eval_env.env.unwrapped.valid_moves() # Access inner env
                if action not in valid_moves:
                    action = np.random.choice(valid_moves) if valid_moves else action

                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated

                if done:
                    # In our wrappers, Positive Reward = Agent Win
                    if reward > 0: wins += 1
                    elif reward < 0: losses += 1
                    else: draws += 1
                    
        print(f"Evaluation: {wins}W / {losses}L / {draws}D")
        return wins / episodes