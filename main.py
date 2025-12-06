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
    
    # Option 1: Manual game type from config
    game_type = config.get('game', {}).get('type', 'perfect')
    
    # Option 2: Load rulebook from file (if specified in config)
    rulebook_file = config.get('game', {}).get('rulebook_file')
    
    if rulebook_file and os.path.exists(rulebook_file):
        print(f"Loading rulebook from: {rulebook_file}")
        with open(rulebook_file, 'r') as f:
            rulebook = f.read()
    else:
        # Option 3: Hardcoded rulebook in code
        rulebook = """Tic-Tac-Toe: Two players alternate placing X and O on a 3x3 grid.
        Player 1 (X) goes first, Player -1 (O) goes second.
        Win condition: Get 3 of your symbols in a row (horizontal, vertical, or diagonal).
        Draw condition: Grid is full with no winner.
        Actions: Choose position 0-8 (row-major order).
        Game state: 3x3 grid with 0=empty, 1=X, -1=O.
        Rewards: +1 for win, -1 for loss, 0 for draw/ongoing."""
    
    print(f"\n[Game Rules]:\n{rulebook}\n")
    print(f"Game Type: {game_type} information\n")
    
    # Select appropriate template based on manual game type
    if game_type == "perfect":
        template_name = "gym_code_perfect"
    elif game_type == "imperfect":
        template_name = "gym_code_imperfect"
    else:
        raise ValueError(f"Invalid game_type: {game_type}. Must be 'perfect' or 'imperfect'")

    # Initial Code Generation with appropriate template
    print("Generating Game Implementation...")
    initial_code = llm.generate_code(template_name, rulebook=rulebook)
    
    # CWM Refinement Loop
    print("Starting CWM Refinement Loop...")
    
    # Pass the template and rulebook to the refiner for error fixing
    final_code = refiner.refine(
        initial_code=initial_code, 
        template=template_name, 
        rulebook=rulebook
    )

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
        # Cleanup: remove the temp file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

    # Proceed with Training using appropriate wrapper for game type
    print("Training PPO agent...")
    trainer = PPOTrainer(env_class, game_type=game_type)
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