from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
import os

class OpenAIClient:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=os.environ.get(config['llm']['api_key_env_var']))
        
        # Store model configurations
        self.creator_model = config['llm']['models']['creator']
        self.refiner_model = config['llm']['models']['refiner']
        
        # Store temperature settings
        self.text_temperature = config['llm']['generation_params']['temperature']
        self.code_temperature = config['llm']['generation_params']['temperature_code']
        
        # Setup Jinja2 environment
        self.env = Environment(loader=FileSystemLoader('prompts'))

    def generate_code(self, template, **kwargs):
        """
        Renders the Jinja2 template and calls OpenAI for code generation.
        Uses creator model and code temperature from config.
        """
        # Load Jinja2 prompt template
        prompt_template = self.env.get_template(f"{template}.jinja2")
        prompt_text = prompt_template.render(**kwargs)

        # API call
        print(f"Calling {self.creator_model} for code generation...")
        response = self.client.chat.completions.create(
            model=self.creator_model,
            messages=[
                {"role": "system", "content": "You are an expert Python RL developer."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=self.code_temperature
        )
        
        content = response.choices[0].message.content
        print("LLM Response:")
        print(content)
        
        # Python code extraction (handles ```python ... ``` and ``` ... ``` cases)
        if "```python" in content:
            content = content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return content

    def generate_text(self, template, **kwargs):
        """
        Renders the Jinja2 template and calls OpenAI for text generation.
        Uses creator model and text temperature from config.
        """
        prompt_template = self.env.get_template(f"{template}.jinja2")
        prompt_text = prompt_template.render(**kwargs)
        
        print(f"Calling {self.creator_model} for text generation...")
        response = self.client.chat.completions.create(
            model=self.creator_model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=self.text_temperature
        )
        return response.choices[0].message.content
    
    def generate_refinement(self, template, **kwargs):
        """
        Renders the Jinja2 template and calls OpenAI for code refinement.
        Uses refiner model and code temperature from config for faster/cheaper refinement.
        """
        prompt_template = self.env.get_template(f"{template}.jinja2")
        prompt_text = prompt_template.render(**kwargs)
        
        print(f"Calling {self.refiner_model} for code refinement...")
        response = self.client.chat.completions.create(
            model=self.refiner_model,
            messages=[
                {"role": "system", "content": "You are an expert Python debugging assistant. Fix the code errors shown."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=self.code_temperature
        )
        
        # Python code extraction (handles ```python ... ``` and ``` ... ``` cases)
        if "```python" in content:
            content = content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return content
    
    # def classify_game(self, rulebook):
    #     """
    #     Classifies a game as 'perfect' or 'imperfect' information.
    #     Uses a classification prompt to determine game type.
    #     """
    #     prompt_template = self.env.get_template("game_classifier.jinja2")
    #     prompt_text = prompt_template.render(rulebook=rulebook)
        
    #     print(f"Classifying game type...")
    #     response = self.client.chat.completions.create(
    #         model=self.refiner_model,  # Use cheaper model for classification
    #         messages=[{"role": "user", "content": prompt_text}],
    #         temperature=0.0  # Deterministic classification
    #     )
        
    #     classification = response.choices[0].message.content.strip().lower()
        
    #     # Validate response
    #     if "perfect" in classification:
    #         game_type = "perfect"
    #     elif "imperfect" in classification:
    #         game_type = "imperfect"
    #     else:
    #         # Default to perfect if unclear
    #         print(f"Warning: Unclear classification '{classification}', defaulting to 'perfect'")
    #         game_type = "perfect"
        
    #     print(f"Game classified as: {game_type} information")
    #     return game_type