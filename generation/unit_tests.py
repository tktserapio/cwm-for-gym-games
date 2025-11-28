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
        
        # Test Reset API
        try:
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                # Old API compatibility - try to handle single return value
                obs = reset_result
                info = {}
        except Exception as e:
            raise ValueError(f"reset() failed: {e}")
        
        if obs is None: 
            raise ValueError("reset() returned None observation")

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
            
            try:
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                elif len(step_result) == 4:
                    # Old API compatibility
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                else:
                    raise ValueError(f"step() must return 4 or 5 values, got {len(step_result)}")
            except Exception as e:
                raise ValueError(f"step() failed: {e}")

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