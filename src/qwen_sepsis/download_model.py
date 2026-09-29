import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from . import MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--local-dir", type=Path, default=Path("models/base/Qwen3-8B"))
    args = parser.parse_args()
    path = snapshot_download(repo_id=args.model, local_dir=args.local_dir)
    print(f"Modelo baixado em: {path}")


if __name__ == "__main__": main()

