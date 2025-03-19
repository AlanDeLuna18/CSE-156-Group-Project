import os
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import json

# Load API key from environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Evaluator:
    def __init__(self, evaluator_model="gpt-4o", temperature=0.0):
        self.evaluator_model = evaluator_model
        self.temperature = temperature
    
    def evaluate_responses(self, dataset_path, output_path, model1_col, model2_col):
        os.makedirs(output_path, exist_ok=True)
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Extract model names
        model1_name = model1_col.replace("_answers", "").lower()
        model2_name = model2_col.replace("_answers", "").lower()
        
        # Create prompt for each question
        results = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating model pairs"):
            question = row['question']
            correct_answer = row['Correct Answer']
            model1_answer = row[model1_col]
            model2_answer = row[model2_col]
            
            # Create evaluation prompt
            prompt = f"""You are evaluating model responses to a technical/academic question.
                    QUESTION:
                    {question}

                    CORRECT ANSWER:
                    {correct_answer}

                    {model1_name.upper()} ANSWER:
                    {model1_answer}

                    {model2_name.upper()} ANSWER:
                    {model2_answer}
                    
                    Evaluate each model answer on these criteria:
                    correctness, reasoning_quality, completeness

                    For "correctness":
                    - 0: Completely wrong answer
                    - 5: Partially correct answer with significant inaccuracies
                    - 8: Answer is correct with very minor imprecisions or omissions
                    - 10: Completely correct answer with perfect accuracy

                    For "reasoning_quality":
                    - 0: Completely flawed reasoning or approach
                    - 3: Contains minimal relevant steps but significant logical errors
                    - 5: Partially correct reasoning with substantial flaws
                    - 8: Sound reasoning with minor errors in execution
                    - 10: Perfect reasoning and approach

                    For "completeness":
                    - 0: Missing critical information or steps
                    - 5: Covers main points but lacks important details
                    - 10: Comprehensive coverage of all relevant information
                   
                    Include a brief explanation for each score in one sentence.
                    Then, determine which model answer is best overall and explain why in one sentence.

                    Format your response as a JSON object with this structure:
                    {{
                      "evaluations": {{
                        "{model1_name}": {{
                          "scores": {{
                            "correctness": score,
                            "reasoning_quality": score,
                            "completeness": score
                          }},
                          "explanations": {{
                            "correctness": "explanation",
                            "reasoning_quality": "explanation",
                            "completeness": "explanation"
                          }}
                        }},
                        "{model2_name}": {{
                          "scores": {{
                            "correctness": score,
                            "reasoning_quality": score,
                            "completeness": score
                          }},
                          "explanations": {{
                            "correctness": "explanation",
                            "reasoning_quality": "explanation",
                            "completeness": "explanation"
                          }}
                        }}
                      }},
                      "best_model": "model_name",
                      "explanation": "reason why this model performed best"
                    }}
                    """
            
            # Make API Call
            messages = [
                {"role": "system", "content": "You are an impartial academic evaluator skilled in technical subjects."},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat.completions.create(
                model=self.evaluator_model,
                messages=messages,
                temperature=self.temperature
            )
            
            evaluation = response.choices[0].message.content.strip()
            
            # Store in results
            try:
                evaluation_data = json.loads(evaluation)
                if "best_model" in evaluation_data:
                    evaluation_data["best_model"] = evaluation_data["best_model"].lower()
                evaluation_data["question_idx"] = i
                evaluation_data["question"] = question
                results.append(evaluation_data)
            except json.JSONDecodeError:
                print(f"Failed to parse evaluation as JSON for question {i}. Using raw text.")
                results.append({
                    "question_idx": i,
                    "question": question,
                    "raw_evaluation": evaluation
                })
        
        # Save results
        results_path = os.path.join(output_path, "comparison_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary
        summary = self._generate_summary(results, model1_name, model2_name)
        summary_path = os.path.join(output_path, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return results, summary
    
    def _generate_summary(self, evaluation_results, model1_name, model2_name):

        criteria = ["correctness", "reasoning_quality", "completeness"]
        
        # Initialize summary
        summary = {
            "models": [model1_name, model2_name],
            "scores": {
                model1_name: {criterion: [] for criterion in criteria},
                model2_name: {criterion: [] for criterion in criteria}
            },
            "wins": {model1_name: 0, model2_name: 0},
            "averages": {}
        }
        
        # Collect scores and count wins
        for result in evaluation_results:
            if "best_model" in result:
                summary["wins"][result["best_model"]] += 1
                
            if "evaluations" in result:
                for model in [model1_name, model2_name]:
                    model_key = model.lower()
                    for criterion in criteria:
                        score = result["evaluations"][model_key]["scores"][criterion]
                        summary["scores"][model_key][criterion].append(score)
        
        # Calculate averages
        for model in [model1_name, model2_name]:
            model_avg_scores = {}
            for criterion in criteria:
                scores = summary["scores"][model][criterion]
                if scores:
                    model_avg_scores[criterion] = sum(scores) / len(scores)
                else:
                    model_avg_scores[criterion] = 0
                    
            summary["averages"][model] = {
                "overall": sum(model_avg_scores.values()) / len(model_avg_scores),
                # "win_percentage": (summary["wins"][model] / len(evaluation_results)) * 100 if evaluation_results else 0
            }
        
        # Determine overall winner
        if summary["averages"][model1_name]["overall"] > summary["averages"][model2_name]["overall"]:
            summary["overall_winner"] = model1_name
        elif summary["averages"][model2_name]["overall"] > summary["averages"][model1_name]["overall"]:
            summary["overall_winner"] = model2_name
        else:
            summary["overall_winner"] = "tie"
        
        return summary

# Example usage
if __name__ == "__main__":
    evaluator = Evaluator()
    results, summary = evaluator.evaluate_responses(
        dataset_path="gpqa_QA_sample.csv",
        output_path="comparison_results",
        model1_col="SelfRefine_answers",
        model2_col="4o_mini_answers"
    )
    
    print(f"Evaluation complete. Results saved to comparison_results/")
    print(f"Overall winner: {summary['overall_winner']}")
    print(f"Average scores:")
    for model in summary["models"]:
        print(f"  {model}: {summary['averages'][model]['overall']:.2f}")