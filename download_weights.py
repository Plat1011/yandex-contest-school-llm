from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-1.7B"
LOCAL_DIR = "weights"

def main() -> None:
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "tokenizer*",
            "*.jinja",
        ],
    )
    print(f"weights downloaded to: {path}")

if __name__ == "__main__":
    main()
