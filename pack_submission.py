import argparse
import zipfile
from pathlib import Path

KEEP_FILES = ["Dockerfile", "solution.py"]
KEEP_WEIGHT_PATTERNS = [
    "*.json", "*.safetensors", "*.txt",
    "tokenizer*", "*.jinja",
]

def collect_weights(weights_dir: Path):
    files = set()
    for pat in KEEP_WEIGHT_PATTERNS:
        for p in weights_dir.glob(pat):
            if p.is_file():
                files.add(p)
    return sorted(files)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights")
    p.add_argument("--out", default="submission.zip")
    args = p.parse_args()

    root = Path(".").resolve()
    weights_dir = Path(args.weights).resolve()
    out_path = Path(args.out).resolve()

    if out_path.exists():
        out_path.unlink()

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as z:
        for name in KEEP_FILES:
            src = root / name
            assert src.exists(), f"missing required file: {src}"
            z.write(src, arcname=name)

        for src in collect_weights(weights_dir):
            arc = Path("weights") / src.name
            z.write(src, arcname=str(arc).replace("\\", "/"))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
