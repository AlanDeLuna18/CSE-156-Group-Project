import time 
import os  
from typing import List, Dict, Any, Optional, Tuple  
from openai import OpenAI  #OpenAI API client
from dotenv import load_dotenv  #For loading environment variables from .env file


#Load API key from environment variables
load_dotenv()  #This loads variables from a .env file into environment variables
api_key = os.getenv("OPENAI_API_KEY")  #Get the OpenAI API key from environment variables
client = OpenAI(api_key=api_key)  #Initialize the OpenAI client with the API key

class SelfRefine:
    """
    A class that manages the self-refinement process using GPT API calls.
    
    The process involves three main steps:
    1. Generate initial text based on a prompt
    2. Get feedback on the generated text
    3. Refine the text based on the feedback
    
    This implements an automated feedback loop where an AI system improves its own output
    through multiple iterations of critique and refinement.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the SelfRefine class with configuration parameters.
        
        Args:
            model (str): The GPT model to use for API calls (default: "gpt-4o-mini")
                         This determines which OpenAI model will generate text, feedback, and refinements
        
        The class maintains several important state variables:
        - current_text: The most recent version of the text
        - feedback_history: A list of all feedback received during refinement
        - refinement_history: A list of all text versions produced during refinement
        - original_prompt: The initial prompt that started the process
        """
        self.model = model  #Store which GPT model to use
        self.current_text = ""  #Will hold the current version of the text
        self.feedback_history = []  #Will store all feedback received during the process
        self.refinement_history = []  #Will store all versions of the text during refinement
        self.original_prompt = ""  #Will store the original prompt
    
    def _make_api_call(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Make an API call to the OpenAI GPT model.
        
        This is a private helper method used by other methods to interact with the OpenAI API.
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the API in the ChatGPT format
                                            Each message has a "role" (system/user/assistant) and "content"
            temperature (float): Controls randomness in the output (0.0-1.0)
                                Lower values make output more deterministic and focused
                                Higher values make output more creative and diverse
            
        Returns:
            str: The response text from the API (stripped of leading/trailing whitespace)
        """
        #Call the OpenAI API with the specified parameters
        response = client.chat.completions.create(
            model=self.model,  #Use the model specified in the constructor
            messages=messages,  #The conversation context to send to the API
            max_tokens=4096,  #Maximum length of the response
            temperature=temperature  #Controls randomness/creativity of the response
        )
        #Extract and return just the text content from the response
        return response.choices[0].message.content.strip()
    
    def generate_initial_text(self, prompt: str) -> str:
        """
        Generate initial text based on a prompt using an API call.
        
        This is the first step in the self-refinement process. It takes a prompt
        and generates an initial response that will later be refined.
        
        Args:
            prompt (str): The prompt to generate text from
                         This is the original request or question
            
        Returns:
            str: The generated text from the model
        """
        #Store the original prompt for later reference during feedback and refinement
        self.original_prompt = prompt
        
        #Prepare the messages for the API call
        #The system message defines the AI's role and behavior
        #The user message contains the actual prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates high-quality text based on prompts."},
            {"role": "user", "content": prompt}
        ]
        
        #Make the API call to generate the initial text
        self.current_text = self._make_api_call(messages)
        #Initialize the refinement history with the initial text
        self.refinement_history = [self.current_text]
        return self.current_text
    
    def get_feedback(self, text: str, temperature: float = 0.7) -> str:
        """
        Get feedback on the current text using an API call, referencing the original prompt.
        
        This is the second step in the self-refinement process. It asks the model to
        critique the current text and suggest improvements.
        
        Args:
            text (str): The text to get feedback on (current version)
            temperature (float): Controls randomness in the feedback
                                Lower values make feedback more consistent
            
        Returns:
            str: The feedback on the text from the model
        """
        #Prepare the messages for the API call
        #The system message instructs the model on how to provide feedback
        messages = [
            {"role": "system", "content": """You are a critical reviewer providing constructive feedback. 
            Analyze the following text based on how well it addresses the original prompt.
            Provide specific, actionable feedback on how it can be improved to better fulfill the requirements
            and intent of the original prompt.
            
            Focus on:
            1. How well the text addresses all aspects of the prompt
            2. Clarity, coherence, and relevance to the prompt
            3. Factual accuracy and overall quality
            4. Any missing information that would better satisfy the prompt
            
            If the text is already factually correct and fully addresses the prompt, clearly state 
            'No further improvements needed' at the beginning of your feedback.
            
            Only suggest changes if they would genuinely improve the text. For factual questions with 
            definitive answers, prioritize accuracy over stylistic changes.
            
            Structure your feedback in bullet points, addressing different aspects of the text.
            """}, 
            
            #The user message provides the original prompt and the text to review
            {"role": "user", "content": 
             f"Original prompt:\n\n{self.original_prompt}\n\nText to review:\n\n{text}\n\nPlease provide detailed feedback on how well this text addresses the original prompt:"}
        ]
        
        #Make the API call to get feedback
        feedback = self._make_api_call(messages, temperature=temperature)
        #Add the feedback to the history
        self.feedback_history.append(feedback)
        return feedback

    def refine_text(self, text: str, feedback: str, temperature: float = 0.7) -> str:
        """
        Refine the text based on feedback using an API call, with reference to the original prompt.
        
        This is the third step in the self-refinement process. It takes the current text and
        the feedback, and produces an improved version of the text.
        
        Args:
            text (str): The text to refine (current version)
            feedback (str): The feedback to incorporate from the previous step
            temperature (float): Controls randomness in the refinement
                                Lower values make refinement more focused on addressing feedback
            
        Returns:
            str: The refined text from the model
        """
        #Prepare the messages for the API call
        #The system message instructs the model on how to refine the text
        messages = [
            {"role": "system", "content": """You are an expert editor. Your task is to improve the given text 
            based on the provided feedback and the original prompt, ONLY if improvements are needed.
            
            If the feedback indicates 'No further improvements needed' or if you determine the text is already 
            optimal, you may return it unchanged. Otherwise, address the feedback points to better fulfill the 
            requirements of the original prompt.
            
            Prioritize factual accuracy over stylistic changes, especially for questions with definitive answers.
            
            Make changes that:
            1. Address valid points in the feedback
            2. Better fulfill the requirements of the original prompt
            3. Ensure the text is fully responsive to what was asked for
            
            If you determine no changes are needed, return the original text.
            """},
            #The user message  provides the original prompt, the text to refine, and the feedback
            {"role": "user", "content": f"Original prompt:\n\n{self.original_prompt}\n\nOriginal text:\n\n{text}\n\nFeedback:\n\n{feedback}\n\nPlease provide a refined version of the text that addresses valid feedback points and better fulfills the original prompt:"}
        ]
        
        #Make the API call to refine the text
        refined_text = self._make_api_call(messages, temperature=temperature)
        #Update the current text with the refined version
        self.current_text = refined_text
        #Add the refined text to the history
        self.refinement_history.append(refined_text)
        return refined_text
    
    def iterate_refinement(self, iterations: int = 3) -> str:
        """
        Perform multiple cycles of feedback and refinement.
        
        This is the main method that orchestrates the entire self-refinement process.
        It repeatedly gets feedback and refines the text for a specified number of iterations
        or until certain stopping conditions are met.
        
        Args:
            iterations (int): The maximum number of refinement iterations to perform
                             Higher values allow for more rounds of improvement
            
        Returns:
            str: The final refined text after all iterations
        """
        for i in range(iterations):
            #Step 1: Get feedback on the current text
            #Use lower temperature for feedback to reduce randomness and get more consistent feedback
            feedback = self.get_feedback(self.current_text, temperature=0.5)

            #Step 2: Check if no improvements are needed
            #If the feedback indicates the text is already optimal, stop the refinement process
            if "no further improvements needed" in feedback.lower():
                print("Model indicates no further refinements are necessary.")
                break
            
            #Save the current text before refinement for comparison and possible reversion
            previous_text = self.current_text
            
            #Step 3: Refine the text based on the feedback
            #Use lower temperature for refinement to reduce randomness and focus on addressing feedback
            self.current_text = self.refine_text(self.current_text, feedback, temperature=0.5)
            
            #Step 4: Check if the text changed after refinement
            #If the text didn't change, it's likely already optimal, so stop the refinement process
            if self.current_text == previous_text:
                print("No changes were made in this refinement iteration. Text is likely optimal.")
                break
            
            #Step 5: Validate the refinement to ensure it improved rather than degraded the text
            #This is a quality control step to prevent the refinement from making the text worse
            validation_messages = [
                {"role": "system", "content": """You are a validation expert. Compare the original and refined texts 
                based on factual accuracy, completeness, and alignment with the original prompt.
                
                Determine if the refinement has improved the text or made it worse.
                
                Return ONLY one of these verdicts:
                "IMPROVED" - if the refinement is better than the original
                "WORSE" - if the refinement introduced errors or is worse than the original
                """},
                {"role": "user", "content": f"Original prompt:\n\n{self.original_prompt}\n\nOriginal text:\n\n{previous_text}\n\nRefined text:\n\n{self.current_text}\n\nIs the refined text better or worse than the original?"}
            ]
            
            #Make the API call to validate the refinement
            #Use very low temperature (0.3) for validation to get more consistent results
            validation = self._make_api_call(validation_messages, temperature=0.3)
            
            #Step 6: Revert to previous version if the refinement made the text worse
            if "worse" in validation.lower():
                print("Refinement validation indicates the changes made the text worse. Reverting to previous version.")
                self.current_text = previous_text
                self.refinement_history.append(previous_text)  #Add the reverted text to history
                break
                    
        #Return the final refined text after all iterations
        return self.current_text


def main(): #example of how to use the SelfRefine class
    """
    Main function to demonstrate and test the self-refinement process.
    
    This function runs the SelfRefine process on a set of test prompts,
    measures performance, and displays results.
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



#Mock Test for unit testing the SelfRefine class
#This is commented out but shows how to test the class with mock API responses
#class TestSelfRefine(unittest.TestCase):
#    @patch.object(SelfRefine, '_make_api_call')
#    def test_full_process(self, mock_api_call):
#        """
#        Test the entire refinement process by mocking API responses at every step.
#        This allows testing without making actual API calls to OpenAI.
#        """
#        prompt = """
#        Write a comprehensive explanation of how neural networks learn through backpropagation.
#        Target audience is undergraduate students with basic knowledge of calculus.
#        """
#        #Mock API responses in sequence:
#        #1. Generate initial text
#        #2. Get feedback (iteration 1)
#        #3. Refine text (iteration 1)
#        #4. Get feedback (iteration 2)
#        mock_api_call.side_effect = [
#            "Initial text: Data preprocessing is important for machine learning because it helps to clean and prepare the data for the algorithm.",
#            "• Clarify the importance of feature scaling.\n• Provide an example.",
#            "Refined text version 1: Data preprocessing is crucial in machine learning because it helps clean the data and prepares it for algorithms. Feature scaling is especially important for algorithms sensitive to input data scale.",
#            "No further improvements needed."
#        ]

#        refiner = SelfRefine(model="gpt-4o-mini")
#        generated_text = refiner.generate_initial_text(prompt)
#        self.assertEqual(generated_text, "Initial text: Data preprocessing is important for machine learning because it helps to clean and prepare the data for the algorithm.")

#        final_text = refiner.iterate_refinement(iterations=3)

#        self.assertEqual(final_text, "Refined text version 1: Data preprocessing is crucial in machine learning because it helps clean the data and prepares it for algorithms. Feature scaling is especially important for algorithms sensitive to input data scale.")
#        self.assertEqual(mock_api_call.call_count, 4) 
#        print(f"Final refined text: {final_text}")



