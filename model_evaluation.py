#!/usr/bin/env python3
"""
Chemical Knowledge Model Evaluation Script

This script evaluates language models using the generated chemical safety questions
from generated_questions.json file.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from dataclasses import dataclass

# Try to import different model inference libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Data class to store evaluation results for a single question"""
    chemical_name: str
    cas_number: str
    ghs_classification: str
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    confidence_score: Optional[float] = None
    response_time: Optional[float] = None
    source_row_index: int = -1


class ModelEvaluator:
    """Main class for evaluating models on chemical safety questions"""
    
    def __init__(self, model_type: str = "transformers", model_name: str = None, **kwargs):
        """
        Initialize the model evaluator
        
        Args:
            model_type: Type of model to use ("openai", "transformers", "api")
            model_name: Name/path of the model
            **kwargs: Additional parameters for model initialization
        """
        self.model_type = model_type
        self.model_name = model_name
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on the specified type"""
        self.logger.info(f"Initializing {self.model_type} model: {self.model_name}")
        
        if self.model_type == "openai" and OPENAI_AVAILABLE:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.model = self.model_name or "gpt-3.5-turbo"
            
        elif self.model_type == "transformers" and TRANSFORMERS_AVAILABLE:
            if not self.model_name:
                raise ValueError("model_name is required for transformers")
            
            self.logger.info("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                **self.kwargs
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        elif self.model_type == "api" and REQUESTS_AVAILABLE:
            self.api_url = self.kwargs.get("api_url")
            if not self.api_url:
                raise ValueError("api_url is required for API model type")
            self.headers = self.kwargs.get("headers", {})
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type} or required libraries not installed")
    
    def load_questions(self, file_path: str) -> List[Dict]:
        """Load questions from the JSON file"""
        self.logger.info(f"Loading questions from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Flatten the questions
        questions = []
        for chemical_data in data:
            if isinstance(chemical_data, dict) and 'questions' in chemical_data:
                for question in chemical_data['questions']:
                    questions.append(question)
        
        self.logger.info(f"Loaded {len(questions)} questions")
        return questions
    
    def format_prompt(self, question: str, chemical_name: str = None) -> str:
        """Format the prompt for the model"""
        prompt_template = """You are an expert in chemical safety. Please answer the following question with either "yes" or "no" only.

Question: {question}

Answer:"""
        
        return prompt_template.format(question=question)
    
    def query_openai_model(self, prompt: str) -> Tuple[str, float]:
        """Query OpenAI model"""
        start_time = time.time()
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            
            answer = response.choices[0].message.content.strip()
            response_time = time.time() - start_time
            
            return answer, response_time
            
        except Exception as e:
            self.logger.error(f"Error querying OpenAI model: {e}")
            return "error", time.time() - start_time
    
    def query_transformers_model(self, prompt: str) -> Tuple[str, float]:
        """Query Transformers model"""
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            response_time = time.time() - start_time
            return answer, response_time
            
        except Exception as e:
            self.logger.error(f"Error querying Transformers model: {e}")
            return "error", time.time() - start_time
    
    def query_api_model(self, prompt: str) -> Tuple[str, float]:
        """Query API model"""
        start_time = time.time()
        
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": 10,
                "temperature": 0.0
            }
            
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            answer = response.json().get("text", "").strip()
            response_time = time.time() - start_time
            
            return answer, response_time
            
        except Exception as e:
            self.logger.error(f"Error querying API model: {e}")
            return "error", time.time() - start_time
    
    def query_model(self, prompt: str) -> Tuple[str, float]:
        """Query the model based on the model type"""
        if self.model_type == "openai":
            return self.query_openai_model(prompt)
        elif self.model_type == "transformers":
            return self.query_transformers_model(prompt)
        elif self.model_type == "api":
            return self.query_api_model(prompt)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def extract_yes_no_answer(self, response: str) -> str:
        """Extract yes/no answer from model response"""
        response = response.lower().strip()
        
        # Clean up response
        response = re.sub(r'[^\w\s]', ' ', response)
        response = ' '.join(response.split())
        
        # Look for yes/no patterns
        if re.search(r'\byes\b', response):
            return "yes"
        elif re.search(r'\bno\b', response):
            return "no"
        else:
            # If neither yes nor no is clearly found, return the cleaned response
            return response
    
    def evaluate_answer(self, expected: str, predicted: str) -> bool:
        """Evaluate if the predicted answer matches the expected answer"""
        expected = expected.lower().strip()
        predicted = self.extract_yes_no_answer(predicted)
        
        return expected == predicted
    
    def evaluate_questions(self, questions: List[Dict], max_questions: int = None) -> List[EvaluationResult]:
        """Evaluate all questions"""
        if max_questions:
            questions = questions[:max_questions]
        
        results = []
        total_questions = len(questions)
        
        self.logger.info(f"Starting evaluation of {total_questions} questions")
        
        for i, question_data in enumerate(questions, 1):
            self.logger.info(f"Processing question {i}/{total_questions}")
            
            # Extract question information
            chemical_name = question_data.get("chemical_name", "")
            cas_number = question_data.get("cas_number", "")
            ghs_classification = question_data.get("ghs_classification", "")
            question = question_data.get("question", "")
            expected_answer = question_data.get("expected_answer", "").lower()
            source_row_index = question_data.get("source_row_index", -1)
            
            # Format prompt and query model
            prompt = self.format_prompt(question, chemical_name)
            model_response, response_time = self.query_model(prompt)
            
            # Extract yes/no answer
            model_answer = self.extract_yes_no_answer(model_response)
            
            # Evaluate correctness
            is_correct = self.evaluate_answer(expected_answer, model_answer)
            
            # Create result object
            result = EvaluationResult(
                chemical_name=chemical_name,
                cas_number=cas_number,
                ghs_classification=ghs_classification,
                question=question,
                expected_answer=expected_answer,
                model_answer=model_answer,
                is_correct=is_correct,
                response_time=response_time,
                source_row_index=source_row_index
            )
            
            results.append(result)
            
            # Log progress
            if i % 10 == 0 or i == total_questions:
                correct_so_far = sum(1 for r in results if r.is_correct)
                accuracy_so_far = correct_so_far / len(results) * 100
                self.logger.info(f"Progress: {i}/{total_questions}, Accuracy so far: {accuracy_so_far:.2f}%")
        
        return results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict:
        """Calculate evaluation metrics"""
        if not results:
            return {}
        
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.is_correct)
        accuracy = correct_answers / total_questions * 100
        
        # Calculate yes/no specific metrics
        yes_questions = [r for r in results if r.expected_answer == "yes"]
        no_questions = [r for r in results if r.expected_answer == "no"]
        
        yes_correct = sum(1 for r in yes_questions if r.is_correct)
        no_correct = sum(1 for r in no_questions if r.is_correct)
        
        yes_accuracy = (yes_correct / len(yes_questions) * 100) if yes_questions else 0
        no_accuracy = (no_correct / len(no_questions) * 100) if no_questions else 0
        
        # Calculate confusion matrix
        tp = sum(1 for r in results if r.expected_answer == "yes" and r.model_answer == "yes")
        tn = sum(1 for r in results if r.expected_answer == "no" and r.model_answer == "no")
        fp = sum(1 for r in results if r.expected_answer == "no" and r.model_answer == "yes")
        fn = sum(1 for r in results if r.expected_answer == "yes" and r.model_answer == "no")
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Average response time
        avg_response_time = sum(r.response_time for r in results if r.response_time) / total_questions
        
        metrics = {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "yes_questions": len(yes_questions),
            "yes_correct": yes_correct,
            "yes_accuracy": yes_accuracy,
            "no_questions": len(no_questions),
            "no_correct": no_correct,
            "no_accuracy": no_accuracy,
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn
            },
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_response_time": avg_response_time
        }
        
        return metrics
    
    def save_results(self, results: List[EvaluationResult], metrics: Dict, output_file: str):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_data = {
            "evaluation_info": {
                "model_type": self.model_type,
                "model_name": self.model_name,
                "timestamp": timestamp,
                "total_questions": len(results)
            },
            "metrics": metrics,
            "detailed_results": []
        }
        
        # Convert results to dict format
        for result in results:
            result_dict = {
                "chemical_name": result.chemical_name,
                "cas_number": result.cas_number,
                "ghs_classification": result.ghs_classification,
                "question": result.question,
                "expected_answer": result.expected_answer,
                "model_answer": result.model_answer,
                "is_correct": result.is_correct,
                "response_time": result.response_time,
                "source_row_index": result.source_row_index
            }
            output_data["detailed_results"].append(result_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, metrics: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print(f"Model Evaluation Summary")
        print("="*60)
        print(f"Model Type: {self.model_type}")
        print(f"Model Name: {self.model_name}")
        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Correct Answers: {metrics['correct_answers']}")
        print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
        print(f"\nBreakdown by Expected Answer:")
        print(f"  'Yes' questions: {metrics['yes_questions']} (Accuracy: {metrics['yes_accuracy']:.2f}%)")
        print(f"  'No' questions: {metrics['no_questions']} (Accuracy: {metrics['no_accuracy']:.2f}%)")
        print(f"\nPerformance Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  True Positive: {cm['true_positive']}")
        print(f"  True Negative: {cm['true_negative']}")
        print(f"  False Positive: {cm['false_positive']}")
        print(f"  False Negative: {cm['false_negative']}")
        print(f"\nAverage Response Time: {metrics['average_response_time']:.3f} seconds")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on chemical safety questions")
    parser.add_argument("--questions_file", type=str, default="generated_questions.json",
                      help="Path to the questions JSON file")
    parser.add_argument("--model_type", type=str, choices=["openai", "transformers", "api"], 
                      default="transformers", help="Type of model to use")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name or path of the model")
    parser.add_argument("--output_file", type=str, 
                      help="Output file for results (auto-generated if not provided)")
    parser.add_argument("--max_questions", type=int,
                      help="Maximum number of questions to evaluate (for testing)")
    parser.add_argument("--api_url", type=str,
                      help="API URL for API model type")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = args.model_name.replace("/", "_").replace("\\", "_")
        args.output_file = f"evaluation_results_{model_name_clean}_{timestamp}.json"
    
    try:
        # Initialize evaluator
        kwargs = {}
        if args.api_url:
            kwargs["api_url"] = args.api_url
        
        evaluator = ModelEvaluator(
            model_type=args.model_type,
            model_name=args.model_name,
            **kwargs
        )
        
        # Load questions
        questions = evaluator.load_questions(args.questions_file)
        
        # Run evaluation
        results = evaluator.evaluate_questions(questions, max_questions=args.max_questions)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(results)
        
        # Save results
        evaluator.save_results(results, metrics, args.output_file)
        
        # Print summary
        evaluator.print_summary(metrics)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
