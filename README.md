## CMeval


- **`download.py`**: download the models mentioned in the paper from Hugging Face into the `model/` directory.
- **`syn.py`**: call an OpenAI-compatible API to answer synthesis questions from a JSON file.
- **`propetry.py`**: call an OpenAI-compatible API to answer property (True/False) questions from `datasets/property.json`.
- **`chemsafe.py`**: call an OpenAI-compatible API to answer safety (True/False) questions from a CSV file in `datasets/`.

### Requirements

- Python 3.8+
- Installed packages:

```bash
pip install openai pandas huggingface_hub
```

### Basic Usage

1. **Configure API client**  
   In `syn.py`, `propetry.py`, and `chemsafe.py`, fill in:
   - `api_key`
   - `base_url`
   - `model` name
   - `input_file` and `output_file` paths

2. **Download models (optional)**  

```bash
cd cmeval
python download.py --list          # show available models
python download.py                 # download all models
python download.py llama3         # download a single model
```

3. **Run evaluation scripts**

```bash
python syn.py
python propetry.py
python chemsafe.py
```

Each script will read questions from the configured input file and save model answers to the configured output file.

