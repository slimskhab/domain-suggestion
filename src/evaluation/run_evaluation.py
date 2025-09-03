"""
LLM-as-a-Judge Evaluation Framework

This module implements automated evaluation using LLM-as-a-judge for domain name quality assessment.
"""

import os
import json
import openai
import anthropic
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import yaml
from datetime import datetime
import time

@dataclass
class EvaluationResult:
    """Represents an evaluation result for a domain name."""
    prompt: str
    generated_domain: str
    business_type: str
    relevance_score: float
    creativity_score: float
    uniqueness_score: float
    safety_score: float
    grammar_score: float
    overall_score: float
    judge_feedback: str
    evaluation_time: float

@dataclass
class ModelEvaluation:
    """Represents evaluation results for a model."""
    model_name: str
    model_version: str
    evaluation_date: str
    total_samples: int
    avg_relevance: float
    avg_creativity: float
    avg_uniqueness: float
    avg_safety: float
    avg_grammar: float
    avg_overall: float
    std_relevance: float
    std_creativity: float
    std_uniqueness: float
    std_safety: float
    std_grammar: float
    std_overall: float

class LLMJudgeEvaluator:
    """LLM-as-a-Judge evaluator for domain name quality assessment."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the evaluator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup API clients
        self.setup_api_clients()
        
        # Evaluation prompts
        self.evaluation_prompts = self.config['evaluation']['prompts']
        self.scoring_scale = self.config['evaluation']['scoring_scale']
        
    def setup_api_clients(self):
        """Setup API clients for different LLM providers."""
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.openai_client = openai
        else:
            self.openai_client = None
            print("⚠️ OpenAI API key not found")
        
        # Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        else:
            self.anthropic_client = None
            print("⚠️ Anthropic API key not found")
    
    def get_judge_model(self) -> str:
        """Get the configured judge model."""
        return self.config['evaluation']['judge_model']
    
    def create_evaluation_prompt(self, prompt: str, generated_domain: str, 
                                business_type: str, metric: str) -> str:
        """Create an evaluation prompt for a specific metric."""
        base_prompt = self.evaluation_prompts[metric]
        
        full_prompt = f"""
You are an expert evaluator of domain names. Please evaluate the following domain name suggestion.

Business Context: {business_type}
Original Request: {prompt}
Generated Domain Name: {generated_domain}

{base_prompt}

Please provide:
1. A score from 1-10 (where 10 is excellent)
2. Brief reasoning for your score

Format your response as:
Score: [number]
Reasoning: [explanation]
"""
        return full_prompt
    
    def evaluate_with_openai(self, prompt: str, model: str = "gpt-4") -> Tuple[float, str]:
        """Evaluate using OpenAI API."""
        try:
            response = self.openai_client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract score and reasoning
            score = self.extract_score_from_response(response_text)
            reasoning = self.extract_reasoning_from_response(response_text)
            
            return score, reasoning
            
        except Exception as e:
            print(f"Error with OpenAI evaluation: {e}")
            return 5.0, "Error in evaluation"
    
    def evaluate_with_anthropic(self, prompt: str, model: str = "claude-3-sonnet-20240229") -> Tuple[float, str]:
        """Evaluate using Anthropic API."""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Extract score and reasoning
            score = self.extract_score_from_response(response_text)
            reasoning = self.extract_reasoning_from_response(response_text)
            
            return score, reasoning
            
        except Exception as e:
            print(f"Error with Anthropic evaluation: {e}")
            return 5.0, "Error in evaluation"
    
    def extract_score_from_response(self, response_text: str) -> float:
        """Extract numerical score from response text."""
        try:
            # Look for "Score: X" pattern
            if "Score:" in response_text:
                score_line = [line for line in response_text.split('\n') if 'Score:' in line][0]
                score = float(score_line.split(':')[1].strip())
                return max(1.0, min(10.0, score))  # Clamp between 1-10
            else:
                # Try to find any number between 1-10
                import re
                numbers = re.findall(r'\b([1-9]|10)\b', response_text)
                if numbers:
                    return float(numbers[0])
                else:
                    return 5.0  # Default score
        except:
            return 5.0  # Default score
    
    def extract_reasoning_from_response(self, response_text: str) -> str:
        """Extract reasoning from response text."""
        try:
            if "Reasoning:" in response_text:
                reasoning_line = [line for line in response_text.split('\n') if 'Reasoning:' in line][0]
                reasoning = reasoning_line.split('Reasoning:')[1].strip()
                return reasoning
            else:
                # Return the full response if no specific reasoning section
                return response_text
        except:
            return "No reasoning provided"
    
    def evaluate_domain_name(self, prompt: str, generated_domain: str, 
                           business_type: str) -> EvaluationResult:
        """Evaluate a single domain name across all metrics."""
        start_time = time.time()
        
        judge_model = self.get_judge_model()
        
        # Evaluate each metric
        metrics = self.config['evaluation']['metrics']
        scores = {}
        feedbacks = {}
        
        for metric in metrics:
            eval_prompt = self.create_evaluation_prompt(prompt, generated_domain, business_type, metric)
            
            if judge_model.startswith("gpt"):
                score, reasoning = self.evaluate_with_openai(eval_prompt, judge_model)
            elif judge_model.startswith("claude"):
                score, reasoning = self.evaluate_with_anthropic(eval_prompt, judge_model)
            else:
                # Default to OpenAI
                score, reasoning = self.evaluate_with_openai(eval_prompt)
            
            scores[metric] = score
            feedbacks[metric] = reasoning
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Calculate overall score (weighted average)
        overall_score = np.mean(list(scores.values()))
        
        # Combine feedback
        combined_feedback = " | ".join([f"{metric}: {feedback}" for metric, feedback in feedbacks.items()])
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            prompt=prompt,
            generated_domain=generated_domain,
            business_type=business_type,
            relevance_score=scores.get('relevance', 5.0),
            creativity_score=scores.get('creativity', 5.0),
            uniqueness_score=scores.get('uniqueness', 5.0),
            safety_score=scores.get('safety', 5.0),
            grammar_score=scores.get('grammar', 5.0),
            overall_score=overall_score,
            judge_feedback=combined_feedback,
            evaluation_time=evaluation_time
        )
    
    def evaluate_model_outputs(self, model_outputs: List[Dict]) -> List[EvaluationResult]:
        """Evaluate multiple model outputs."""
        results = []
        
        for i, output in enumerate(model_outputs):
            print(f"Evaluating output {i+1}/{len(model_outputs)}...")
            
            result = self.evaluate_domain_name(
                prompt=output['prompt'],
                generated_domain=output['generated_domain'],
                business_type=output['business_type']
            )
            
            results.append(result)
        
        return results
    
    def create_evaluation_summary(self, results: List[EvaluationResult], 
                                model_name: str, model_version: str) -> ModelEvaluation:
        """Create a summary of evaluation results."""
        if not results:
            return None
        
        # Calculate statistics
        relevance_scores = [r.relevance_score for r in results]
        creativity_scores = [r.creativity_score for r in results]
        uniqueness_scores = [r.uniqueness_score for r in results]
        safety_scores = [r.safety_score for r in results]
        grammar_scores = [r.grammar_score for r in results]
        overall_scores = [r.overall_score for r in results]
        
        return ModelEvaluation(
            model_name=model_name,
            model_version=model_version,
            evaluation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=len(results),
            avg_relevance=np.mean(relevance_scores),
            avg_creativity=np.mean(creativity_scores),
            avg_uniqueness=np.mean(uniqueness_scores),
            avg_safety=np.mean(safety_scores),
            avg_grammar=np.mean(grammar_scores),
            avg_overall=np.mean(overall_scores),
            std_relevance=np.std(relevance_scores),
            std_creativity=np.std(creativity_scores),
            std_uniqueness=np.std(uniqueness_scores),
            std_safety=np.std(safety_scores),
            std_grammar=np.std(grammar_scores),
            std_overall=np.std(overall_scores)
        )
    
    def save_evaluation_results(self, results: List[EvaluationResult], 
                              summary: ModelEvaluation, filename: str = None):
        """Save evaluation results to file."""
        os.makedirs("data/evaluation_data", exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = f"data/evaluation_data/{filename}"
        
        # Convert results to dictionaries
        results_dict = []
        for result in results:
            results_dict.append({
                'prompt': result.prompt,
                'generated_domain': result.generated_domain,
                'business_type': result.business_type,
                'relevance_score': result.relevance_score,
                'creativity_score': result.creativity_score,
                'uniqueness_score': result.uniqueness_score,
                'safety_score': result.safety_score,
                'grammar_score': result.grammar_score,
                'overall_score': result.overall_score,
                'judge_feedback': result.judge_feedback,
                'evaluation_time': result.evaluation_time
            })
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump({
                'summary': summary.__dict__,
                'results': results_dict
            }, f, indent=2)
        
        print(f"✅ Evaluation results saved to {filepath}")
        return filepath

def main():
    """Main function to run evaluation."""
    print("🚀 Starting LLM-as-a-Judge evaluation...")
    
    # Initialize evaluator
    evaluator = LLMJudgeEvaluator()
    
    # Example evaluation (you would typically load model outputs here)
    example_outputs = [
        {
            'prompt': 'Generate a domain name for a technology startup.',
            'generated_domain': 'techflow.com',
            'business_type': 'tech_startup'
        },
        {
            'prompt': 'Create a website domain for a restaurant.',
            'generated_domain': 'tastebuds.com',
            'business_type': 'restaurant'
        }
    ]
    
    # Evaluate outputs
    results = evaluator.evaluate_model_outputs(example_outputs)
    
    # Create summary
    summary = evaluator.create_evaluation_summary(results, "example_model", "v1.0")
    
    # Save results
    evaluator.save_evaluation_results(results, summary)
    
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()
