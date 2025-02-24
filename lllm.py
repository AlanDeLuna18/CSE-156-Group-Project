import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class SelfRefine:
    """
    A class that manages the self-refinement process using GPT API calls.
    
    The process involves three main steps:
    1. Generate initial text based on a prompt
    2. Get feedback on the generated text
    3. Refine the text based on the feedback
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the SelfRefine class.
        
        Args:
            model (str): The GPT model to use for API calls
        """
        self.model = model
        self.current_text = ""
    
    def _make_api_call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Make an API call to the OpenAI GPT model.
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the API
            temperature (float): The temperature parameter for the API call
            
        Returns:
            str: The response from the API
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=16000,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    def generate_initial_text(self, prompt: str) -> str:
        """
        Generate initial text based on a prompt using an API call.
        
        Args:
            prompt (str): The prompt to generate text from
            
        Returns:
            str: The generated text
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates high-quality text based on prompts."},
            {"role": "user", "content": prompt}
        ]
        
        self.current_text = self._make_api_call(messages)
        return self.current_text
    
    def get_feedback(self, text: str) -> str:
        """
        Get feedback on the current text using an API call.
        
        Args:
            text (str): The text to get feedback on
            
        Returns:
            str: The feedback on the text
        """
        messages = [
            {"role": "system", "content": """You are a critical reviewer providing constructive feedback. 
            Analyze the following text and provide specific, actionable feedback on how it can be improved.
            Focus on clarity, coherence, factual accuracy, and overall quality.
            Be detailed and point out specific issues along with suggestions for improvement.
            Structure your feedback in bullet points, addressing different aspects of the text."""},
            
            {"role": "user", "content": f"Please provide detailed feedback on the following text:\n\n{text}"}
        ]
        
        return self._make_api_call(messages, temperature=0.7)

    def refine_text(self, text: str, feedback: str) -> str:
        """
        Refine the text based on feedback using an API call.
        
        Args:
            text (str): The text to refine
            feedback (str): The feedback to incorporate
            
        Returns:
            str: The refined text
        """
        pass
    
    def iterate_refinement(self, iterations: int = 3) -> str:
        """
        Perform multiple cycles of feedback and refinement.
        
        Args:
            iterations (int): The number of refinement iterations to perform
            
        Returns:
            str: The final refined text
        """
        pass


def main():
    """
    Main function to demonstrate the self-refinement process.
    """
    refiner = SelfRefine(model="gpt-4o-mini")
    
    prompt = """
    Write a comprehensive explanation of how neural networks learn through backpropagation.
    Target audience is undergraduate students with basic knowledge of calculus.
    """
    
    initial_text = refiner.generate_initial_text(prompt)
    print(initial_text)


if __name__ == "__main__":
    main()
