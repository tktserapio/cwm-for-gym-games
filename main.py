import os
import sys
import time
import yaml
import gymnasium as gym
import inspect
import traceback
import importlib.util
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
load_dotenv()

from generation.cwm_refiner import CWMRefiner
from generation.llm_client import OpenAIClient
from validation.ppo_trainer import PPOTrainer

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Set up LLM and Refiner
    llm = OpenAIClient(config)
    refiner = CWMRefiner(llm_client=llm, max_attempts=config['refinement']['max_attempts'])
    
    # Generate Game Rulebook
    print("Generating Novel Game Rules...")
    rulebook = llm.generate_text("game_design")
    print(f"\n[Generated Game Rules]:\n{rulebook}\n")

    # Initial Code Generation
    # Uses prompts/gym_code.jinja2
    print("Generating Game Implementation...")
    initial_code = llm.generate_code("gym_code", rulebook=rulebook)
    
    # CWM Refinement Loop
    # Run unit test trajectories and prompt LLM to fix errors iteratively
    print("Starting CWM Refinement Loop...")
    final_code = refiner.refine(initial_code=initial_code)

    if not final_code:
        print("Failed to generate valid code within the refinement budget.")
        return

    # Save the generated code to a temporary file
    temp_file_name = "temp_generated_env.py"
    with open(temp_file_name, "w") as f:
        f.write(final_code)

    try:
        # Load the module from the file path
        spec = importlib.util.spec_from_file_location("temp_generated_env", temp_file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_generated_env"] = module
        spec.loader.exec_module(module)
        
        # Look for CustomEnv
        env_class = getattr(module, "CustomEnv", None)
        
        if env_class is None:
            # Fallback search in the module
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, gym.Env) and obj is not gym.Env:
                    env_class = obj
                    break
        
        if env_class is None:
            raise ValueError("No valid Gym environment found in generated code.")

        print(f"Successfully loaded class: {env_class.__name__}")

    except Exception as e:
        print(f"Error: Could not import generated code.\nError: {e}")
        return
    finally:
        # Cleanup: remove the temp file if you want
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

    # Proceed with Training
    print("Training PPO agent...")
    trainer = PPOTrainer(env_class)
    win_rate = trainer.train_and_evaluate(total_timesteps=config['rl_training']['total_timesteps'])
    
    print(f"FINAL RESULTS")
    print(f"Agent Win Rate: {win_rate * 100:.1f}%")
    
    # Win thresholds (from gg-bench paper)
    min_wr = config['rl_training']['validation']['min_win_rate']
    max_wr = config['rl_training']['validation']['max_win_rate']
    
    if min_wr < win_rate < max_wr:
        print("Game is balanced and playable.")

        # Save final code to output
        timestamp = int(time.time())
        game_name = f"generated_game_{timestamp}.py"
        output_path = os.path.join(config['project']['output_dir'], game_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(final_code)
        print(f"Saved to: {output_path}")
        
    else:
        print(f"Game rejected (Winrate outside {min_wr}-{max_wr} range).")

if __name__ == "__main__":
    main()