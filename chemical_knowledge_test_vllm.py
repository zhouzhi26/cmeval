#!/usr/bin/env python3
#测newchemical_questions_en
"""
Qwen2.5-7B Chemical Knowledge Test with vLLM Acceleration
High-performance batch testing using vLLM for faster inference
"""

import json
import time
import signal
import sys
import os
from datetime import datetime
from pathlib import Path
import threading
import queue
import asyncio
from typing import List, Dict, Any


class ChemicalKnowledgeTesterVLLM:
    def __init__(self, model_path="models/Meta-Llama-3.1-8B-Instruct"):
        self.model_path = model_path
        self.llm = None
        self.results = []
        self.current_question_index = 0
        self.total_questions = 0
        self.start_time = None
        self.results_file = None
        self.incorrect_answers_file = None
        self.timeout_seconds = 30
        self.batch_size = 16  # vLLM batch processing
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "timeout_errors": 0,
            "generation_errors": 0,
            "start_time": None,
            "chemicals_processed": set()
        }
        
        # Signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal"""
        print(f"\n⚠️  Interrupt signal received, saving current progress...")
        self._save_final_results()
        sys.exit(0)
    
    def load_vllm_model(self):
        """Load model with vLLM"""
        print("🚀 Loading model with vLLM acceleration...")
        
        try:
            from vllm import LLM, SamplingParams
            
            # vLLM configuration for optimal performance
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=1,  # Adjust based on available GPUs
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                trust_remote_code=True,
                dtype="bfloat16",
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192
            )
            
            # Sampling parameters for deterministic and accurate answers
            self.sampling_params = SamplingParams(
                temperature=0.1,        # Low temperature for more deterministic responses
                max_tokens=10,          # Sufficient tokens for clear "yes" or "no" responses
                top_p=0.9,             # High top_p for good vocabulary coverage
                repetition_penalty=1.0, # No repetition penalty to avoid distortion
                stop=["</s>", "\n\n"]   # Standard stop tokens
            )
            
            print("✅ vLLM model loaded successfully!")
            return True
            
        except ImportError:
            print("❌ vLLM not installed. Please install: pip install vllm")
            return False
        except Exception as e:
            print(f"❌ vLLM model loading failed: {e}")
            return False
    
    def create_prompt(self, question_text: str) -> str:
        """Create English prompt for chemical questions"""
        prompt = f""" Answer the following chemistry-related yes/no question. 
Respond with only "Yes" or "No", no explanation needed.

Question: {question_text}

Answer:"""
        return prompt
    
    def create_chat_prompt(self, question_text: str) -> str:
        """Create clear and direct prompt for accurate yes/no answers"""
        
        # Clear, unbiased system instruction
        chat_prompt = "<|im_start|>system\nYou are a knowledgeable chemistry expert. Answer chemistry questions accurately with only 'Yes' or 'No'. Use your scientific knowledge to provide correct answers.<|im_end|>\n"
        chat_prompt += f"<|im_start|>user\nQuestion: {question_text}\n\nAnswer (Yes or No):<|im_end|>\n"
        chat_prompt += "<|im_start|>assistant\n"
        
        return chat_prompt
    
    def parse_answer(self, response: str) -> bool:
        """Parse model response to boolean answer"""
        response_lower = response.lower().strip()
        
        # Clean up common response patterns
        response_lower = response_lower.replace("answer:", "").strip()
        response_lower = response_lower.replace("the answer is", "").strip()
        response_lower = response_lower.replace(".", "").strip()
        response_lower = response_lower.replace("!", "").strip()
        
        # Look for yes/no in the response
        if "yes" in response_lower and "no" not in response_lower:
            return True
        elif "no" in response_lower and "yes" not in response_lower:
            return False
        elif response_lower.startswith("yes"):
            return True
        elif response_lower.startswith("no"):
            return False
        else:
            # If unclear, return the original response for debugging
            return response.strip()
    
    def process_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """Process a batch of questions using vLLM"""
        # Prepare prompts
        prompts = []
        for item in batch_data:
            question_text = item["question_data"]["question"]
            prompt = self.create_chat_prompt(question_text)
            prompts.append(prompt)
        
        # Generate responses
        try:
            start_time = time.time()
            outputs = self.llm.generate(prompts, self.sampling_params)
            end_time = time.time()
            
            batch_processing_time = end_time - start_time
            
            # Process results
            results = []
            for i, output in enumerate(outputs):
                item = batch_data[i]
                chemical_data = item["chemical_data"]
                question_data = item["question_data"]
                
                response_text = output.outputs[0].text if output.outputs else ""
                qwen_answer = self.parse_answer(response_text)
                correct_answer = question_data["answer"]
                
                result = {
                    "chemical_name": chemical_data.get("chemical_name", ""),
                    "cas_number": chemical_data.get("cas_number", ""),
                    "ghs_classification": chemical_data.get("ghs_codes", chemical_data.get("ghs_classification", "")),
                    "question_id": question_data["question_id"],
                    "question": question_data["question"],
                    "correct_answer": correct_answer,
                    "qwen_answer": qwen_answer,
                    "qwen_raw_response": response_text,
                    "is_correct": None,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": batch_processing_time / len(batch_data)
                }
                
                # Calculate accuracy - convert both to boolean for comparison
                if isinstance(qwen_answer, bool):
                    # Convert correct_answer to boolean for comparison
                    correct_answer_bool = correct_answer.lower().strip() == "yes" if isinstance(correct_answer, str) else correct_answer
                    result["is_correct"] = (qwen_answer == correct_answer_bool)
                    if result["is_correct"]:
                        self.stats["correct_answers"] += 1
                    else:
                        self.stats["incorrect_answers"] += 1
                else:
                    result["status"] = "parse_error"
                    result["is_correct"] = False
                    self.stats["generation_errors"] += 1
                
                self.stats["total_processed"] += 1
                self.stats["chemicals_processed"].add(chemical_data["chemical_name"])
                
                results.append(result)
            
            return results
            
        except Exception as e:
            # Handle batch processing errors
            error_results = []
            for item in batch_data:
                chemical_data = item["chemical_data"]
                question_data = item["question_data"]
                
                result = {
                    "chemical_name": chemical_data.get("chemical_name", ""),
                    "cas_number": chemical_data.get("cas_number", ""),
                    "ghs_classification": chemical_data.get("ghs_codes", chemical_data.get("ghs_classification", "")),
                    "question_id": question_data["question_id"],
                    "question": question_data["question"],
                    "correct_answer": question_data["answer"],
                    "qwen_answer": None,
                    "qwen_raw_response": f"Error: {str(e)}",
                    "is_correct": False,
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": 0
                }
                
                self.stats["generation_errors"] += 1
                self.stats["total_processed"] += 1
                error_results.append(result)
            
            return error_results
    
    def save_results_batch(self, results: List[Dict]):
        """Save batch results to file"""
        if self.results_file:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
    
    def print_batch_progress(self, results: List[Dict]):
        """Print progress for batch processing"""
        accuracy = (self.stats["correct_answers"] / max(1, self.stats["total_processed"])) * 100
        elapsed_time = time.time() - self.start_time
        avg_time_per_question = elapsed_time / max(1, self.stats["total_processed"])
        eta = avg_time_per_question * (self.total_questions - self.current_question_index)
        
        # Print batch summary
        correct_in_batch = sum(1 for r in results if r.get("is_correct"))
        batch_accuracy = (correct_in_batch / len(results)) * 100
        
        print(f"\n📦 Batch [{self.current_question_index-len(results)+1}-{self.current_question_index}] completed:")
        print(f"   🎯 Batch accuracy: {batch_accuracy:.1f}% ({correct_in_batch}/{len(results)})")
        print(f"   📈 Overall accuracy: {accuracy:.1f}% ({self.stats['correct_answers']}/{self.stats['total_processed']})")
        print(f"   ⏱️  Estimated remaining time: {eta/60:.1f} minutes")
        print(f"   🧪 Chemicals processed: {len(self.stats['chemicals_processed'])}")
        
        # Print individual results
        for result in results:
            status_icon = "✅" if result.get("is_correct") else "❌" if result.get("is_correct") is False else "⚠️"
            print(f"   {status_icon} {result['chemical_name']} Q{result['question_id']}: "
                  f"{result['correct_answer']} → {result['qwen_answer']}")
    
    def load_questions(self, json_file):
        """Load questions from JSON file"""
        print(f"📖 Loading questions from: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Count total questions
            total_questions = 0
            for chemical in data:
                total_questions += len(chemical.get("questions", []))
            
            self.total_questions = total_questions
            print(f"📊 Loaded {len(data)} chemicals with {total_questions} questions")
            return data
            
        except Exception as e:
            print(f"❌ Failed to load questions: {e}")
            return None
    
    def run_test(self, json_file, output_file=None, max_questions=None):
        """Run batch testing with vLLM"""
        # Set output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"chemical_test_vllm_results_{timestamp}.jsonl"
        
        self.results_file = output_file
        
        # Clear/create output file
        with open(self.results_file, 'w', encoding='utf-8') as f:
            pass
        
        print(f"💾 Results will be saved to: {self.results_file}")
        
        # Load vLLM model
        if not self.load_vllm_model():
            return False
        
        # Load questions
        chemical_data = self.load_questions(json_file)
        if not chemical_data:
            return False
        
        # Start testing
        self.start_time = time.time()
        self.stats["start_time"] = datetime.now().isoformat()
        
        print(f"\n🧪 Starting chemical knowledge test with vLLM acceleration...")
        print(f"⚡ Batch size: {self.batch_size}")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if max_questions:
            print(f"🎯 Max questions: {max_questions}")
        print("=" * 70)
        
        try:
            question_count = 0
            batch_data = []
            
            for chemical in chemical_data:
                questions = chemical.get("questions", [])
                
                for question in questions:
                    if max_questions and question_count >= max_questions:
                        break
                    
                    # Add to batch
                    batch_data.append({
                        "chemical_data": chemical,
                        "question_data": question
                    })
                    
                    question_count += 1
                    
                    # Process batch when full
                    if len(batch_data) >= self.batch_size:
                        results = self.process_batch(batch_data)
                        self.save_results_batch(results)
                        self.current_question_index = question_count
                        self.print_batch_progress(results)
                        batch_data = []
                
                if max_questions and question_count >= max_questions:
                    break
            
            # Process remaining batch
            if batch_data:
                results = self.process_batch(batch_data)
                self.save_results_batch(results)
                self.current_question_index = question_count
                self.print_batch_progress(results)
        
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            
        finally:
            self._save_final_results()
    
    def _save_final_results(self):
        """Save final statistics"""
        total_time = time.time() - self.start_time if self.start_time else 0
        accuracy = (self.stats["correct_answers"] / max(1, self.stats["total_processed"])) * 100
        
        final_stats = {
            "test_summary": {
                "engine": "vLLM",
                "batch_size": self.batch_size,
                "total_questions_processed": self.stats["total_processed"],
                "correct_answers": self.stats["correct_answers"],
                "incorrect_answers": self.stats["incorrect_answers"],
                "timeout_errors": self.stats["timeout_errors"],
                "generation_errors": self.stats["generation_errors"],
                "accuracy_percentage": accuracy,
                "total_time_seconds": total_time,
                "questions_per_second": self.stats["total_processed"] / max(1, total_time),
                "avg_time_per_question": total_time / max(1, self.stats["total_processed"]),
                "chemicals_processed": len(self.stats["chemicals_processed"]),
                "unique_chemicals": list(self.stats["chemicals_processed"]),
                "start_time": self.stats["start_time"],
                "end_time": datetime.now().isoformat()
            }
        }
        
        # Save statistics
        if self.results_file:
            stats_file = self.results_file.replace('.jsonl', '_summary.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
            
            # Save incorrect answers
            if self.incorrect_answers_file:
                incorrect_answers_file = self.incorrect_answers_file
            else:
                incorrect_answers_file = self.results_file.replace('.jsonl', '_incorrect_answers.json')
            self.save_incorrect_answers(incorrect_answers_file)
        
        # Print final results
        print("\n" + "=" * 70)
        print("🎉 vLLM Test Completed! Final Statistics:")
        print(f"   ⚡ Engine: vLLM (batch size: {self.batch_size})")
        print(f"   ✅ Correct answers: {self.stats['correct_answers']}")
        print(f"   ❌ Incorrect answers: {self.stats['incorrect_answers']}")
        print(f"   🔧 Generation errors: {self.stats['generation_errors']}")
        print(f"   📈 Overall accuracy: {accuracy:.2f}%")
        print(f"   🧪 Chemicals processed: {len(self.stats['chemicals_processed'])}")
        print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"   🚀 Speed: {self.stats['total_processed'] / max(1, total_time):.1f} questions/second")
        print(f"   💾 Results file: {self.results_file}")
        print(f"   📋 Summary file: {stats_file}")

    def save_incorrect_answers(self, output_file: str):
        """Save incorrectly answered questions to a separate file"""
        incorrect_questions = []
        
        # Read all results from the results file
        if self.results_file and os.path.exists(self.results_file):
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line.strip())
                        if result.get("is_correct") is False:  # Only incorrect answers
                            incorrect_question = {
                                "chemical_name": result.get("chemical_name", ""),
                                "cas_number": result.get("cas_number", ""),
                                "question": result.get("question", ""),
                                "answer": result.get("correct_answer", "")
                            }
                            incorrect_questions.append(incorrect_question)
        
        if incorrect_questions:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(incorrect_questions, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(incorrect_questions)} incorrect answers to {output_file}")
        else:
            print("ℹ️  No incorrect answers to save")
        
        return len(incorrect_questions)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chemical Knowledge Test with vLLM")
    parser.add_argument("--input", default="datasets/generated_questions.json", help="Input JSON file")
    parser.add_argument("--output", help="Output JSONL file (auto-generated if not specified)")
    parser.add_argument("--incorrect-answers-file", help="Output file for incorrect answers (auto-generated if not specified)")
    parser.add_argument("--model-path", default="models/Meta-Llama-3.1-8B-Instruct", help="Model path")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions for testing")
    parser.add_argument("--batch-size", type=int, default=8, help="vLLM batch size")
    
    args = parser.parse_args()
    
    print("⚡ Chemical Knowledge Test with vLLM")
    print("=" * 50)
    
    # Check input file
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Create tester
    tester = ChemicalKnowledgeTesterVLLM(model_path=args.model_path)
    tester.batch_size = args.batch_size
    
    # Set incorrect answers file if specified
    if args.incorrect_answers_file:
        tester.incorrect_answers_file = args.incorrect_answers_file
    
    # Run test
    tester.run_test(
        json_file=args.input,
        output_file=args.output,
        max_questions=args.max_questions
    )


if __name__ == "__main__":
    main()
