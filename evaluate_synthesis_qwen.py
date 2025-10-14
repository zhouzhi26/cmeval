#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a Qwen model with vLLM on tagged_synthesis_smile.json
- Split SMILES questions and non-SMILES questions
- Batch inference using vLLM
- Save outputs to hecheng/smile/qwen3-8b_eval.json and hecheng/common/qwen3-8b_eval.json

Options:
- --mode {both,smile,common} to choose which subset to evaluate (default: both)

This follows the style of qwen3.py for constructing prompts and batched generation,
but writes per-item detailed outputs compatible with existing result files.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
import argparse


# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-1___7B"
DATASET_PATH = "/root/autodl-tmp/tagged_synthesis_smile.json"
# Use qwen2.5_7b.json only as format reference; write to new eval files
OUTPUT_FILE_SMILE = "/root/autodl-tmp/hecheng/smile/qwen3-1_7b_eval.json"
OUTPUT_FILE_COMMON = "/root/autodl-tmp/hecheng/common/qwen3-1_7b_eval.json"

BATCH_SIZE = 2
MAX_MODEL_LEN = 30000
GPU_MEM_UTIL = 0.9

# Decoding similar to qwen3.py
sampling_params = SamplingParams(
    temperature=0.1,  # Refer to qwen3.py, lower temperature and use prompts to suppress "thinking" output
    top_p=0.9,
    max_tokens=512,
)


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_rows_with_question(rows: List[Dict[str, Any]], question_key: str) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        q = r.get(question_key)
        if isinstance(q, str) and q.strip():
            filtered.append(r)
    return filtered


def build_prompts(rows: List[Dict[str, Any]], question_key: str) -> List[str]:
    prompts: List[str] = []
    for r in rows:
        q = r.get(question_key, "").strip()
        # Disable "deep thinking/chain-of-thought", only output final steps and conclusions (refer to qwen3.py approach)
        prompt = (
            "You are a chemistry assistant. Do not include any chain-of-thought, hidden reasoning, thinking steps, or internal analysis. "
            "Provide only the final actionable synthesis steps and concise conclusions.\n"
            f"Question: {q}\n"
            "Answer:"
        )
        prompts.append(prompt)
    return prompts


def generate(llm: LLM, prompts: List[str]) -> List[str]:
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        result = llm.generate(batch, sampling_params)
        for out in result:
            text = out.outputs[0].text if out.outputs else ""
            outputs.append(text)
    return outputs


def write_outputs(rows: List[Dict[str, Any]], responses: List[str], evaluated_key: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    now = datetime.now().isoformat()

    detailed: List[Dict[str, Any]] = []
    for idx, (row, resp) in enumerate(zip(rows, responses)):
        item = {
            "chemistry_name": row.get("chemistry_name"),
            "smile_name": row.get("smile_name"),
            "chemistry_name_question": row.get("chemistry_name_question"),
            "smile_name_question": row.get("smile_name_question"),
            "source_tag": row.get("source_tag"),
            # Match existing files' response placement based on evaluated type
            ("smile_name_response" if evaluated_key == "smile_name_question" else "chemistry_name_response"): resp.strip(),
            "evaluated_question_type": ("smile_name" if evaluated_key == "smile_name_question" else "chemistry_name"),
            "evaluation_timestamp": now,
            "evaluation_index": idx,
            "model_config": {
                "model_path": MODEL_PATH,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEM_UTIL,
                "batch_size": BATCH_SIZE,
            },
        }
        detailed.append(item)

    # Existing files appear to be plain arrays of detailed items (no top-level metrics)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-8B on tagged_synthesis_smile.json with vLLM",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "smile", "common"],
        default="both",
        help="Select which subset to evaluate: both (default), smile, or common.",
    )
    args = parser.parse_args()

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} items from {DATASET_PATH}")

    # Independent filtering: separately count samples with respective question fields
    smile_rows = filter_rows_with_question(dataset, "smile_name_question")
    common_rows = filter_rows_with_question(dataset, "chemistry_name_question")
    print(f"SMILES questions: {len(smile_rows)}, Common questions: {len(common_rows)}")

    # Initialize LLM
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
    )

    # SMILES block
    if args.mode in ("both", "smile") and smile_rows:
        smile_prompts = build_prompts(smile_rows, "smile_name_question")
        smile_responses = generate(llm, smile_prompts)
        write_outputs(smile_rows, smile_responses, "smile_name_question", OUTPUT_FILE_SMILE)
        print(f"Wrote SMILES outputs to {OUTPUT_FILE_SMILE}")

    # Common block
    if args.mode in ("both", "common") and common_rows:
        common_prompts = build_prompts(common_rows, "chemistry_name_question")
        common_responses = generate(llm, common_prompts)
        write_outputs(common_rows, common_responses, "chemistry_name_question", OUTPUT_FILE_COMMON)
        print(f"Wrote Common outputs to {OUTPUT_FILE_COMMON}")

    print("All done.")


if __name__ == "__main__":
    main()


