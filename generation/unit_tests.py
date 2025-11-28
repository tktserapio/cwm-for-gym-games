import gymnasium as gym
import numpy as np
import importlib.util
import sys
import os
import inspect

def run_trajectory_test(code_str: str) -> bool:
    """
    Saves code to a temp file and imports it to verify API compliance.
    """
    temp_file_name = "temp_cwm_test_env.py"
    
    try:
        with open(temp_file_name, "w") as f:
            f.write(code_str)

        spec = importlib.util.spec_from_file_location("temp_cwm_test_env", temp_file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_cwm_test_env"] = module
        spec.loader.exec_module(module)

        env_class = getattr(module, "CustomEnv", None)
        if env_class is None:
            # Fallback search
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, gym.Env) and obj is not gym.Env:
                    env_class = obj
                    break
        
        if env_class is None:
            raise ValueError("No class inheriting from gym.Env found.")

        env = env_class()
        
        # Test Reset
        obs, info = env.reset()
        if obs is None: 
            raise ValueError("reset() returned None")

        # Test Random Walk
        # We simulate a 'smart' random agent that only picks valid moves
        # to ensure the game logic itself holds up.
        for _ in range(15):
            valid_moves = env.valid_moves()
            if not isinstance(valid_moves, list):
                 raise ValueError(f"valid_moves() must return a list, got {type(valid_moves)}")
            
            if len(valid_moves) == 0:
                # If no moves, game must be over
                if not (env.unwrapped.terminated or env.unwrapped.truncated):
                    raise ValueError("No valid moves returned in non-terminal state")
                break

            action = np.random.choice(valid_moves)
            obs, reward, terminated, truncated, info = env.step(action)

            if not isinstance(reward, (int, float)):
                 raise ValueError(f"Reward must be number, got {type(reward)}")

            if terminated or truncated:
                env.reset()
                
        return True

    except Exception as e:
        print(f"UNIT TEST FAILED: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)