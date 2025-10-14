#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Qwen3-8B on chemical Q&A dataset using vLLM.
- Force disable thinking mode (enable_thinking=False)
- Model outputs only "true" / "false"
- Batch inference
- Calculate Accuracy / Precision / Recall / F1
- Save as JSON file (with detailed results)
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vllm import LLM, SamplingParams

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "/root/autodl-tmp/models/Qwen/Qwen3-8B"  # Qwen3-8B downloaded from ModelScope or HuggingFace
INPUT_FILE = "output.json"            # Input question file
OUTPUT_FILE = "qwen3_8b.json"           # Save result file
BATCH_SIZE = 2           # Batch size

# -----------------------------
# Load test set
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset: List[Dict] = json.load(f)

total_questions = len(dataset)
print(f"Loaded {total_questions} questions.")

# -----------------------------
# Initialize vLLM
# -----------------------------
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="bfloat16",
    max_num_batched_tokens=4096)

# Force disable thinking mode to ensure only final answer output
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=5,  # Only output true/false
)

# -----------------------------
# Build prompts
# -----------------------------
prompts = []
meta_data = []

for item in dataset:
    question = item["question"]
    # Disable thinking mode -> explicitly require in prompt
    prompt = (
        "You must answer only with 'true' or 'false'. "
        "Do not include any reasoning or thinking steps.\n"
        f"Question: {question}\nAnswer:"
    )
    prompts.append(prompt)
    meta_data.append(item)

# -----------------------------
# Batch inference
# -----------------------------
results = []

for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    batch_meta = meta_data[i:i+BATCH_SIZE]

    outputs = llm.generate(batch_prompts, sampling_params)

    for out, meta in zip(outputs, batch_meta):
        raw_resp = out.outputs[0].text.strip()
        model_resp = raw_resp.lower()

        # Only keep true/false
        if "true" in model_resp:
            model_resp = "true"
        elif "false" in model_resp:
            model_resp = "false"
        else:
            model_resp = "false"  # Default fallback

        correct = (model_resp == meta["answer"].lower())

        results.append({
            "chemical_name": meta["chemical_name"],
            "cas_number": meta["cas_number"],
            "question": meta["question"],
            "answer": meta["answer"],
            "model_response": model_resp,
            "raw_response": raw_resp,
            "correct": correct,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": MODEL_NAME
        })

# -----------------------------
# Calculate metrics
# -----------------------------
y_true = [item["answer"].lower() for item in dataset]
y_pred = [r["model_response"].lower() for r in results]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="true")
recall = recall_score(y_true, y_pred, pos_label="true")
f1 = f1_score(y_true, y_pred, pos_label="true")

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

# -----------------------------
# Save JSON
# -----------------------------
output = {
    "model_info": {
        "model_name": MODEL_NAME,
        "dataset_name": os.path.basename(INPUT_FILE).split(".")[0],
        "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_questions
    },
    "metrics": metrics,
    "detailed_results": results
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Evaluation finished. Results saved to {OUTPUT_FILE}")
