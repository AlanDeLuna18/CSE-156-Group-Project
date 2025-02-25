import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import unittest
from unittest.mock import patch

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
        self.feedback_history = []
        self.refinement_history = []
    
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
            max_tokens=1024,
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
        self.refinement_history = [self.current_text]
        return self.current_text
    
    def get_feedback(self, text: str, temperature: float = 0.7) -> str:
        """
        Get feedback on the current text using an API call.
        
        Args:
            text (str): The text to get feedback on
            temperature (float): The temperature parameter for the API call
            
        Returns:
            str: The feedback on the text
        """
        messages = [
            {"role": "system", "content": """You are a critical reviewer providing constructive feedback. 
            Analyze the following text and provide specific, actionable feedback on how it can be improved.
            Focus on clarity, coherence, factual accuracy, and overall quality.
            Be detailed and point out specific issues along with suggestions for improvement.
            Structure your feedback in bullet points, addressing different aspects of the text.
            
            IMPORTANT: Only include the phrase 'No further improvements needed' if the text is truly optimal 
            and you cannot identify ANY areas for improvement, no matter how small.
            
            You MUST find at least 3 specific areas for improvement, no matter how good the text is.
            """}, 
            
            {"role": "user", "content": f"Please provide detailed feedback on the following text:\n\n{text}"}
        ]
        
        feedback = self._make_api_call(messages, temperature=temperature)
        self.feedback_history.append(feedback)
        return feedback

    def refine_text(self, text: str, feedback: str, temperature: float = 0.7) -> str:
        """
        Refine the text based on feedback using an API call.
        
        Args:
            text (str): The text to refine
            feedback (str): The feedback to incorporate
            temperature (float): The temperature parameter for the API call
            
        Returns:
            str: The refined text
        """
        messages = [
            {"role": "system", "content": """You are an expert editor. Your task is to improve the given text 
            based on the provided feedback. Make all necessary changes to address the feedback points while 
            preserving the original intent and information. 
            
            IMPORTANT: You MUST make meaningful changes to the text based on the feedback.
            Even if the changes are minor, ensure you address each point of feedback.
            Provide a complete, refined version of the text that is clearly different from the original.
            
            DO NOT return the text unchanged - this is critical.
            """},
            {"role": "user", "content": f"Original text:\n\n{text}\n\nFeedback:\n\n{feedback}\n\nPlease provide a refined version of the text that addresses all the feedback points:"}
        ]
        
        refined_text = self._make_api_call(messages, temperature=temperature)
        self.current_text = refined_text
        self.refinement_history.append(refined_text)
        return refined_text
    
    def iterate_refinement(self, iterations: int = 3) -> str:
        """
        Perform multiple cycles of feedback and refinement.
        
        Args:
            iterations (int): The number of refinement iterations to perform
            
        Returns:
            str: The final refined text
        """
        for i in range(iterations):
            feedback = self.get_feedback(self.current_text)
            print(f"Feedback from iteration {i+1}:\n{feedback}\n")

            if i > 0 and "no further improvements needed" in feedback.lower():
                print("Model indicates no further refinements are necessary.")
                break
            
            previous_text = self.current_text
            
            self.current_text = self.refine_text(self.current_text, feedback)
            
            if self.current_text == previous_text:
                print("Warning: No changes were made in this refinement iteration. Trying again with stronger feedback.")
                
                stronger_feedback = feedback + "\n\nADDITIONAL FEEDBACK: The text needs significant improvement. Please address ALL the points above and make substantial changes to enhance clarity, coherence, and overall quality."
                
                self.current_text = self.refine_text(previous_text, stronger_feedback, temperature=0.9)
            
            print(f"Iteration {i+1} complete.\n")
        
        return self.current_text

def main():
    """
    Main function to demonstrate and test the self-refinement process.
    """
    test_prompts = [
        {
            "name": "Neural Networks",
            "prompt": """
            Write a concise explanation of how neural networks learn through backpropagation.
            Target audience is undergraduate students with basic knowledge of calculus.
            Keep it under 300 words and focus on the key concepts.
            """,
            "iterations": 2
        },
        {
            "name": "Climate Change",
            "prompt": """
            Explain the causes and effects of climate change in a way that a high school student would understand.
            Include 3 key scientific concepts and 3 potential solutions.
            Keep it under 400 words and use simple language.
            """,
            "iterations": 2
        }
    ]
    
    for test_case in test_prompts:
        print(f"\n{'='*80}\nTESTING: {test_case['name']}\n{'='*80}")
        
        refiner = SelfRefine(model="gpt-4o-mini")
        
        start_time = time.time()
        
        print(f"\n[1] Generating initial text for prompt: {test_case['name']}")
        initial_text = refiner.generate_initial_text(test_case['prompt'])
        print(f"\nINITIAL TEXT:\n{initial_text}\n")
        
        print(f"\n[2] Performing {test_case['iterations']} refinement iterations")
        final_text = refiner.iterate_refinement(iterations=test_case['iterations'])
        
        execution_time = time.time() - start_time
        
        print(f"\nFINAL REFINED TEXT:\n{final_text}\n")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        initial_word_count = len(initial_text.split())
        final_word_count = len(final_text.split())
        word_count_change = final_word_count - initial_word_count
        
        print(f"\nTEST RESULTS:")
        print(f"- Initial word count: {initial_word_count}")
        print(f"- Final word count: {final_word_count}")
        print(f"- Word count change: {word_count_change:+d}")
        print(f"- Improvement iterations: {test_case['iterations']}")
        
        if initial_text == final_text:
            print("- Status: NO CHANGE (The text remained the same after refinement)")
        else:
            print("- Status: IMPROVED (The text was successfully refined)")
            
        print("\nREFINEMENT HISTORY:")
        for i, version in enumerate(refiner.refinement_history):
            if i == 0:
                print(f"Version {i}: Initial Text ({len(version.split())} words)")
            else:
                word_diff = len(version.split()) - len(refiner.refinement_history[i-1].split())
                print(f"Version {i}: Refinement {i} ({len(version.split())} words, {word_diff:+d} words)")


if __name__ == "__main__":
    main()

    # Mock Test
    # unittest.main()


# Mock Test
# class TestSelfRefine(unittest.TestCase):
#     @patch.object(SelfRefine, '_make_api_call')
#     def test_full_process(self, mock_api_call):
#         """
#         Test the entire refinement process by mocking API responses at every step.
#         """
#         prompt = """
#         Write a comprehensive explanation of how neural networks learn through backpropagation.
#         Target audience is undergraduate students with basic knowledge of calculus.
#         """
#         # Mock API responses in sequence:
#         # 1. Generate initial text
#         # 2. Get feedback (iteration 1)
#         # 3. Refine text (iteration 1)
#         # 4. Get feedback (iteration 2)
#         mock_api_call.side_effect = [
#             "Initial text: Data preprocessing is important for machine learning because it helps to clean and prepare the data for the algorithm.",
#             "• Clarify the importance of feature scaling.\n• Provide an example.",
#             "Refined text version 1: Data preprocessing is crucial in machine learning because it helps clean the data and prepares it for algorithms. Feature scaling is especially important for algorithms sensitive to input data scale.",
#             "No further improvements needed."
#         ]

#         refiner = SelfRefine(model="gpt-4o-mini")
#         generated_text = refiner.generate_initial_text(prompt)
#         self.assertEqual(generated_text, "Initial text: Data preprocessing is important for machine learning because it helps to clean and prepare the data for the algorithm.")

#         final_text = refiner.iterate_refinement(iterations=3)

#         self.assertEqual(final_text, "Refined text version 1: Data preprocessing is crucial in machine learning because it helps clean the data and prepares it for algorithms. Feature scaling is especially important for algorithms sensitive to input data scale.")
#         self.assertEqual(mock_api_call.call_count, 4) 
#         print(f"Final refined text: {final_text}")



