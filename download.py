
from huggingface_hub import snapshot_download
from pathlib import Path


MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B-Instruct",
    "phi4": "microsoft/Phi-4",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma3": "google/gemma-2-9b-it",
}

DOWNLOAD_DIR = Path(__file__).parent / "model"


def download(model_key=None, token=None):

    if model_key and model_key not in MODELS:
        print(f"Error'{model_key}'")
        print(f"Model: {', '.join(MODELS.keys())}")
        return
    
    models_to_download = {model_key: MODELS[model_key]} if model_key else MODELS
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    for key, repo_id in models_to_download.items():
        print(f"Downloading: {key} ({repo_id})")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(DOWNLOAD_DIR / key),
                token=token,
                resume_download=True
            )
            print(f"✓ {key} Downloaded\n")
        except Exception as e:
            print(f"✗ {key} Download failed: {e}\n")


if __name__ == "__main__":
    import sys
    
    if "--list" in sys.argv:
        print("Available models:")
        for key, repo_id in MODELS.items():
            print(f"  {key}: {repo_id}")
    else:
        model_key = sys.argv[1] if len(sys.argv) > 1 else None
        token = None
        for arg in sys.argv:
            if arg.startswith("--token="):
                token = arg.split("=")[1]
        download(model_key, token)
