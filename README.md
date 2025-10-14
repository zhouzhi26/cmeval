# CMEval - Chemical Knowledge Evaluation for Small Language Models

CMEval is a comprehensive evaluation toolkit specifically designed for assessing the performance of **Small Language Models (SLMs)** in the chemical knowledge domain. The project supports various evaluation methods optimized for smaller models, including batch testing and vLLM-accelerated inference.

## Key Features

- ✅ **Small Model Optimized**: Designed specifically for evaluating small language models (0.5B-8B parameters)
- ⚡ **High-Performance Inference**: Uses vLLM for efficient batch-accelerated inference on resource-constrained environments
- 📊 **Comprehensive Metrics**: Provides multiple evaluation metrics including accuracy, precision, recall, and F1 score
- 🔄 **Batch Evaluation**: Supports automated batch evaluation of multiple small models
- 💾 **Detailed Results**: Saves complete test results and error analysis
- 🚀 **Resource Efficient**: Optimized for models that can run on consumer-grade GPUs (16GB+ VRAM)

## Project Structure

```
cmeval/
├── chemical_knowledge_test.py        # Basic chemical knowledge test script
├── chemical_knowledge_test_vllm.py   # vLLM-accelerated version of chemical knowledge test
├── batch_model_evaluation.py         # Batch model evaluation tool
├── model_evaluation.py               # General model evaluation framework
├── qwen3.py                          # Qwen3 small model-specific evaluation script
├── evaluate_synthesis_qwen.py        # Chemical synthesis evaluation (supports SMILES format)
├── datasets/                         # Dataset directory
├── models/                           # Model directory
└── results/                          # Evaluation results output directory
```

## Installation

### System Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- At least 16GB VRAM (recommended for models up to 8B parameters)
- 8GB VRAM minimum (for models up to 3B parameters)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Single Model Evaluation

Use the basic version for chemical knowledge testing:

```bash
python chemical_knowledge_test.py \
    --model-path /path/to/your/model \
    --dataset-path /path/to/dataset.json \
    --output results/output.json
```

### 2. vLLM Accelerated Evaluation

Use vLLM for high-performance batch inference (recommended for small models):

```bash
python chemical_knowledge_test_vllm.py \
    --input /path/to/dataset.json \
    --output results/output.jsonl \
    --model-path /path/to/your/model \
    --batch-size 16
```

### 3. Batch Model Evaluation

Automatically evaluate multiple small models:

```bash
python batch_model_evaluation.py \
    --dataset /path/to/dataset.json \
    --output-dir results/ \
    --no-skip
```

Note: You need to configure the list of model paths in `batch_model_evaluation.py`.

### 4. Qwen3 Small Model Evaluation

Evaluation specifically for Qwen3 small model series:

```bash
python qwen3.py
```

Configuration parameters can be modified in the configuration section at the top of the script.

### 5. Chemical Synthesis Evaluation

Evaluate model performance on chemical synthesis problems (supports SMILES format):

```bash
python evaluate_synthesis_qwen.py --mode both
```

Available modes:
- `both`: Evaluate both SMILES and common chemical questions
- `smile`: Evaluate only SMILES questions
- `common`: Evaluate only common chemical questions

## Dataset Format

### Chemical Knowledge Q&A Dataset

```json
[
  {
    "chemical_name": "Chemical substance name",
    "cas_number": "CAS number",
    "question": "Question content",
    "answer": "true/false or yes/no"
  }
]
```

### Chemical Synthesis Dataset

```json
[
  {
    "chemistry_name": "Chemical substance name",
    "smile_name": "SMILES format",
    "chemistry_name_question": "Common chemical question",
    "smile_name_question": "SMILES format question",
    "source_tag": "Data source tag"
  }
]
```

## Evaluation Metrics

All evaluation scripts calculate the following metrics:

- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of actual positives among predicted positives
- **Recall**: Proportion of correctly predicted actual positives
- **F1 Score**: Harmonic mean of precision and recall

## Output Results

Evaluation results are saved in JSON format, including:

```json
{
  "model_info": {
    "model_name": "Model name",
    "dataset_name": "Dataset name",
    "test_timestamp": "Test time",
    "total_questions": 1000
  },
  "metrics": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "f1_score": 0.85
  },
  "detailed_results": [
    {
      "question": "Question content",
      "answer": "Correct answer",
      "model_response": "Model response",
      "correct": true
    }
  ]
}
```

## Configuration

### Batch Evaluation Configuration

Modify the model list in `batch_model_evaluation.py`:

```python
self.models = [
    "/root/autodl-tmp/models/Qwen/Qwen3-0___6B",
    "/root/autodl-tmp/models/Qwen/Qwen3-1___7B",
    "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    "/root/autodl-tmp/models/Qwen/Qwen3-8B"
]
```

### vLLM Configuration Parameters

```python
# Inference parameters optimized for small models
temperature=0.1          # Lower temperature for more deterministic answers
max_tokens=20           # Maximum number of tokens to generate
top_p=0.9               # Nucleus sampling parameter
batch_size=16           # Batch processing size (adjust based on model size)
gpu_memory_utilization=0.9  # GPU memory utilization rate
```

## Supported Small Language Models

- ✅ **Qwen Series** (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B)
- ✅ **Llama Series** (Llama-3.2-1B, Llama-3.2-3B, Meta-Llama-3.1-8B-Instruct)
- ✅ **Phi Series** (Phi-2, Phi-3-mini, Phi-3-small)
- ✅ **Gemma Series** (Gemma-2B, Gemma-7B)
- ✅ **Other Transformers-based small models** (typically 0.5B-8B parameters)
- ✅ **Custom fine-tuned small models**

## Common Issues

### 1. Out of Memory

If you encounter out-of-memory issues with small models, you can:
- Reduce `batch_size` (try 8, 4, or 2)
- Decrease `max_model_len` (try 2048 or 1024 for smaller models)
- Lower `gpu_memory_utilization` (try 0.8 or 0.7)
- Use model quantization (int8 or int4)

### 2. Qwen3 Model Thinking Mode Issue

Qwen3 small models may output thinking processes. The project disables this through:
- Explicitly requiring no thinking process in prompts
- Using specific stop tokens: `<|im_end|>`, `</think>`, `<think>`
- Setting shorter `max_tokens` limits

### 3. Answer Parsing Failures

If model output cannot be correctly parsed as True/False or Yes/No, the script will:
- Try multiple parsing strategies (contains word, starts with word, etc.)
- Record the original response for debugging
- Output warning messages in logs

### 4. Small Model Performance

For optimal performance with small models:
- Use simpler, more direct prompts
- Reduce context length when possible
- Consider fine-tuning for domain-specific tasks
- Use quantization for faster inference

## Performance Optimization

### Using vLLM for Small Models

vLLM provides significant performance improvements for small models:
- **Batch processing**: Process multiple questions simultaneously
- **PagedAttention**: Optimize memory usage (crucial for small GPUs)
- **Continuous batching**: Automatically optimize batch scheduling
- **Quantization support**: Run larger small models on limited hardware

### Batch Evaluation

`batch_model_evaluation.py` supports:
- Automatically skip completed evaluations (disable with `--no-skip`)
- Save and resume evaluation progress
- Detailed progress reports and time estimates
- Efficient resource allocation for sequential model testing

### Memory Optimization Tips

For running multiple small models efficiently:
- Use `torch.cuda.empty_cache()` between model loads
- Enable gradient checkpointing for larger models
- Use mixed precision (fp16/bf16) when supported
- Consider using CPU offloading for very large batches

## Development Guide

### Adding New Small Model Support

1. Add model path in the corresponding evaluation script
2. If special handling is needed, refer to model detection logic in `qwen3.py`
3. Adjust prompt format to adapt to the new model's instruction format
4. Test with smaller batch sizes first to ensure stability

### Custom Evaluation Metrics

Add new metric calculation logic in the `calculate_metrics()` function.

### Extending Dataset Format

Modify the `load_questions()` or corresponding data loading function.

### Fine-tuning Small Models

To fine-tune small models on chemical knowledge:
1. Use the evaluation results to identify weak areas
2. Prepare domain-specific training data
3. Use parameter-efficient fine-tuning (LoRA, QLoRA)
4. Re-evaluate using this toolkit

## License

This project is licensed under the MIT License.

## Contributing

Issues and Pull Requests are welcome! We especially welcome:
- Support for new small language models
- Optimization techniques for resource-constrained environments
- Chemical domain-specific evaluation datasets

## Contact

If you have any questions or suggestions, please contact us through Issues.

## Citation

If you use CMEval in your research, please cite:

```bibtex
@misc{cmeval2024,
  title={CMEval: Chemical Knowledge Evaluation for Small Language Models},
  author={CMEval Contributors},
  year={2024},
  howpublished={\url{https://github.com/your-repo/cmeval}}
}
```

---

**Note**: This tool is designed for research and evaluation purposes, specifically focusing on small language models (0.5B-8B parameters). Please comply with the usage agreements of relevant models.
