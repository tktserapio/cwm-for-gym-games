import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class SelfPlayWrapper(gym.Wrapper):
    """
    Wrapper to enable self-play training.
    CRITICAL: Canonicalizes the observation so the Agent always sees itself as '1'.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.agent_player = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Randomly assign agent to play as player 1 (X) or -1 (O)
        self.agent_player = np.random.choice([1, -1])
        info['agent_player'] = self.agent_player
        
        # If agent is player -1 (O), the Opponent (X) must move first
        if self.agent_player == -1:
            valid_moves = self.env.unwrapped.valid_moves()
            if valid_moves:
                # Random opponent move
                action = np.random.choice(valid_moves)
                obs, _, _, _, info = self.env.step(action)
        
        # ### FIX 1: Canonicalize observation (Flip board if playing as -1)
        # Now Agent always sees '1' as self, even if it's actually '-1' on the board
        return obs * self.agent_player, info
    
    def step(self, action):
        # 1. Agent makes its move
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # [CRITICAL FIX] Terminate immediately on invalid move
        # This prevents the agent from accumulating -100 penalties that hide the win signal.
        if reward == -10:
            terminated = True
            return obs * self.agent_player, reward, terminated, truncated, info

        # 2. If game not over, Opponent makes move (Random Policy)
        if not (terminated or truncated):
            if self.env.unwrapped.current_player != self.agent_player:
                valid_moves = self.env.unwrapped.valid_moves()
                if valid_moves:
                    opponent_action = np.random.choice(valid_moves)
                    obs, opp_reward, terminated, truncated, info = self.env.step(opponent_action)
                    
                    # Flip reward: If opponent won (opp_reward=1), Agent gets -1
                    if terminated or truncated:
                        reward = -opp_reward
                    else:
                        reward = 0

        return obs * self.agent_player, reward, terminated, truncated, info

class PPOTrainer:
    def __init__(self, env_class):
        self.env_class = env_class

    def train_and_evaluate(self, total_timesteps=100_000):
        # Vectorize environment with the CORRECTED wrapper
        vec_env = DummyVecEnv([lambda: SelfPlayWrapper(self.env_class())])

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4, 
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            batch_size=64,
            n_steps=2048,
            verbose=0
        )

        try:
            print(f"Training PPO for {total_timesteps} steps...")
            model.learn(total_timesteps=total_timesteps)
            
            win_rate = self._evaluate_winrate(model)
            return win_rate
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _evaluate_winrate(self, model, episodes=100):
        env = self.env_class()
        wins = 0
        
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            
            # Alternate agent player
            agent_player = 1 if ep % 2 == 0 else -1
            
            # If Agent is O, Opponent (X) moves first
            if agent_player == -1:
                 valid_moves = env.unwrapped.valid_moves()
                 action = np.random.choice(valid_moves)
                 obs, _, done, _, _ = env.step(action)

            while not done:
                # ### FIX 3: Flip Obs during Evaluation too!
                canonical_obs = obs * agent_player
                action, _ = model.predict(canonical_obs, deterministic=True)
                
                # Fallback for illegal moves
                valid_moves = env.unwrapped.valid_moves()
                if action not in valid_moves:
                    action = np.random.choice(valid_moves) if valid_moves else action

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if done:
                    # If Agent moved and won, reward is +1
                    if reward == 1: wins += 1
                    break

                # Opponent Turn
                valid_moves = env.unwrapped.valid_moves()
                if valid_moves:
                    opp_action = np.random.choice(valid_moves)
                    obs, reward, terminated, truncated, _ = env.step(opp_action)
                    done = terminated or truncated
                    
                    if done:
                        # If Opponent moved and won, reward is +1 (for opponent)
                        # So for agent, this is a loss.
                        break
                    
        return wins / episodes