#!/usr/bin/env python3
# Test incorrect answers
import json
import time
import os
from pathlib import Path
from vllm import LLM, SamplingParams
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer

# Configuration parameters
MODEL_PATH = "/root/autodl-tmp/models/llama/Meta-Llama-3___1-8B-Instruct"
JSON_PATH = "/root/autodl-tmp/output.json"
BATCH_SIZE = 16  # Adjust based on GPU memory

# Extract model name from model path
MODEL_NAME = os.path.basename(MODEL_PATH)
# Extract dataset name from dataset path
DATASET_NAME = os.path.splitext(os.path.basename(JSON_PATH))[0]
# Generate output filename
OUTPUT_PATH = f"{MODEL_NAME}_{DATASET_NAME}.json"

# Global variables, will be initialized in main function
llm = None
tokenizer = None
is_qwen3_model = False

# Define generation parameters - consistent with vLLM version
sampling_params = SamplingParams(
    temperature=0.1,        # Low temperature for more deterministic responses
    max_tokens=20,           # Reduced tokens for shorter "True" or "False" responses
    top_p=0.9,             # High top_p for good vocabulary coverage
    repetition_penalty=1.0, # No repetition penalty to avoid distortion  # More stop tokens to prevent extra text
)

# Token parameters for Qwen3 model
qwen3_sampling_params = SamplingParams(
    temperature=0.1,        # Low temperature for more deterministic responses
    max_tokens=100,         # Increase Qwen3 token count to ensure complete answer generation
    top_p=0.9,             # High top_p for good vocabulary coverage
    repetition_penalty=1.0, # No repetition penalty to avoid distortion
    stop=["<|im_end|>", "\n", "</think>", "<think>"]  # Add stop tokens
)

def create_prompt(question, chemical_name, cas_number, is_qwen3=False):
    """Create prompt format consistent with vLLM version"""
    if is_qwen3:
        # For Qwen3 model, create message format for tokenizer.apply_chat_template
        messages = [
            {"role": "user", "content": f"Question: {question}\n\nAnswer with only True or False. Do NOT use <think> tags. Do NOT think step by step. Answer directly with True or False only. /no_think"}
        ]
        return messages
    else:
        # For other models, use original vLLM format
        chat_prompt = "system\nYou are a knowledgeable chemistry expert. You must answer chemistry questions with ONLY 'True' or 'False'. Do not provide any explanations, reasoning, or additional text.\n"
        chat_prompt += f"user\nQuestion: {question}\n\nAnswer with only True or False:\n"
        chat_prompt += "assistant\n"
        return chat_prompt

def parse_answer(response: str) -> str:
    """Parse model response, accurately identify True/False answers"""
    response_lower = response.lower().strip()
    
    # Clean up common response patterns
    response_lower = response_lower.replace("answer:", "").strip()
    response_lower = response_lower.replace("the answer is", "").strip()
    response_lower = response_lower.replace("the answer:", "").strip()
    response_lower = response_lower.replace("answer is", "").strip()
    response_lower = response_lower.replace(".", "").strip()
    response_lower = response_lower.replace("!", "").strip()
    response_lower = response_lower.replace(",", "").strip()
    
    # More precise answer detection logic
    # 1. Check if contains explicit True/False
    if "true" in response_lower and "false" not in response_lower:
        return "True"
    elif "false" in response_lower and "true" not in response_lower:
        return "False"
    
    # 2. Check if starts with True/False
    if response_lower.startswith("true"):
        return "True"
    elif response_lower.startswith("false"):
        return "False"
    
    # 3. Check if contains other words representing True/False
    true_indicators = ["yes", "correct", "right", "1", "positive", "affirmative"]
    false_indicators = ["no", "incorrect", "wrong", "0", "negative"]
    
    # Check True indicators
    for indicator in true_indicators:
        if indicator in response_lower:
            # Ensure no False indicators
            has_false = any(false_ind in response_lower for false_ind in false_indicators)
            if not has_false:
                return "True"
    
    # Check False indicators
    for indicator in false_indicators:
        if indicator in response_lower:
            # Ensure no True indicators
            has_true = any(true_ind in response_lower for true_ind in true_indicators)
            if not has_true:
                return "False"
    
    # 4. If no match, return original response for debugging
    print(f"Warning: Unable to parse answer from response: '{response}'")
    print(f"   Cleaned response: '{response_lower}'")
    return response.strip()
   

def process_batch(batch, is_qwen3=False):
    """Process a single batch and return results"""
    if is_qwen3:
        # For Qwen3 model, try using vLLM directly instead of tokenizer.apply_chat_template
        # Because enable_thinking=False parameter doesn't work in current version
        print("🔄 Using vLLM for Qwen3 model due to enable_thinking=False not working")
        
        # Convert message format to vLLM-compatible prompt format
        vllm_prompts = []
        for messages in batch:
            # Extract question content
            question = messages[0]['content'].split('Question: ')[1].split(' /no_think')[0]
            # Create vLLM format prompt
            vllm_prompt = f"system\nYou are a knowledgeable chemistry expert. You must answer chemistry questions with ONLY 'True' or 'False'. Do not provide any explanations, reasoning, or additional text.\nuser\nQuestion: {question}\n\nAnswer with only True or False:\nassistant\n"
            vllm_prompts.append(vllm_prompt)
        
        # Use vLLM to generate responses
        outputs = llm.generate(vllm_prompts, qwen3_sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]
    else:
        # For other models, use vLLM
        outputs = llm.generate(batch, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

def calculate_metrics(y_true, y_pred):
    """Calculate various evaluation metrics"""
    # Convert True/False to 1/0
    y_true_binary = [1 if label.lower() == 'true' else 0 for label in y_true]
    y_pred_binary = [1 if label.lower() == 'true' else 0 for label in y_pred]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Calculate precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Chemical Knowledge Test with vLLM")
    parser.add_argument("--dataset-path", default=JSON_PATH, help="Dataset JSON file path")
    parser.add_argument("--output", help="Output JSON file (auto-generated if not specified)")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Model path")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions for testing")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="vLLM batch size")
    
    args = parser.parse_args()
    
    # Extract model name from model path
    model_name = os.path.basename(args.model_path)
    # Extract dataset name from dataset path
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        output_path = f"{model_name}_{dataset_name}.json"
    
    # Update model path and dataset path
    model_path = args.model_path
    json_path = args.dataset_path
    batch_size = args.batch_size
    
    print("⚡ Chemical Knowledge Test with vLLM")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_path}")
    
    # Initialize model (using new model path)
    print(f"🔄 Initializing model with: {model_path}")
    try:
        global llm, tokenizer, is_qwen3_model
        
        # Detect if it's a Qwen3 model - refer to tokenizer.apply_chat_template parameter setting
        is_qwen3 = any(keyword in model_name.lower() for keyword in ["qwen3", "qwen-3", "qwen_3"])
        is_qwen3_model = is_qwen3
        
        if is_qwen3:
            # For Qwen3 model, also use vLLM because tokenizer.apply_chat_template's enable_thinking=False doesn't work
            print("🔍 Detected Qwen3 model - using vLLM with Qwen3-specific configurations")
            # Base LLM parameters
            llm_params = {
                "model": model_path,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 4096
            }
            
            llm = LLM(**llm_params)
            print("✅ Qwen3 vLLM model initialized successfully")
        else:
            # For other models, use vLLM
            print("🔍 Using vLLM for non-Qwen3 model")
            # Base LLM parameters
            llm_params = {
                "model": model_path,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 4096
            }
            
            llm = LLM(**llm_params)
            print("✅ vLLM model initialized successfully")
            
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Check dataset file
    if not Path(json_path).exists():
        print(f"❌ Dataset file not found: {json_path}")
        return
    
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Limit question count
    if args.max_questions and args.max_questions < len(data):
        data = data[:args.max_questions]
        print(f"📊 Limited to {args.max_questions} questions")

    # Batch processing
    results = []
    y_true = []  # True labels
    y_pred = []  # Predicted labels
    
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        
        # Create prompts
        prompts = []
        for item in batch_data:
            prompt = create_prompt(
                item["question"], 
                item["chemical_name"], 
                item["cas_number"],
                is_qwen3=is_qwen3  # Pass Qwen3 detection result
            )
            prompts.append(prompt)
        
        # Generate responses
        responses = process_batch(prompts, is_qwen3=is_qwen3)
        
        # Record results
        for j, (item, response) in enumerate(zip(batch_data, responses)):
                # Use answer parsing consistent with vLLM version
                clean_response = parse_answer(response)
                
                # Normalize answer format: convert to lowercase for comparison
                clean_response_normalized = clean_response.lower()
                expected_answer_normalized = item["answer"].lower()
                
                # Collect labels for metric calculation
                y_true.append(item["answer"])
                y_pred.append(clean_response_normalized)
                
                record = {
                    **item,
                    "model_response": clean_response_normalized,
                    "raw_response": response,
                    "correct": clean_response_normalized == expected_answer_normalized,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model_name
                }
                results.append(record)
                print(f"Processed {i+j+1}/{len(data)} | Question: {item['question'][:50]}... | Expected: {item['answer']} | Got: {clean_response_normalized} | Correct: {record['correct']}")

    # Calculate evaluation metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create final result structure
    final_results = {
        "model_info": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(results)
        },
        "metrics": {
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score']
        },
        "detailed_results": results
    }
    
    # Print final results
    print(f"\n" + "="*60)
    print(f"Final Results:")
    print(f"="*60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Total questions: {len(results)}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"="*60)

    # Save results to a single JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()