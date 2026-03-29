#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import time

# ===============================
# Paths
# ===============================
HOME_DIR = Path.home()             # /opt/app-root/src in S2I
APP_DIR = HOME_DIR                 # repo root
MODELS_DIR = HOME_DIR / "models"
QWEN_MODEL = MODELS_DIR / "Qwen3-8B"
EMBED_MODEL = MODELS_DIR / "nomic-embed-text-v1.5"
LOG_FILE = HOME_DIR / "cpu-qwen3-app.log"

# ===============================
# Ensure models directory exists
# ===============================
MODELS_DIR.mkdir(exist_ok=True)

# ===============================
# Helper to clone repo if not exists
# ===============================
def clone_if_missing(repo_url, target_path):
    if target_path.exists():
        print(f"{target_path} already exists, skipping clone")
        return
    print(f"Cloning {repo_url} into {target_path}...")
    subprocess.run(["git", "clone", repo_url, str(target_path)], check=True)
    # Git LFS pull (if enabled)
    subprocess.run(["git", "-C", str(target_path), "lfs", "pull"], check=False)
    print(f"Cloned {repo_url} successfully")

# ===============================
# Clone required models
# ===============================
clone_if_missing("https://huggingface.co/Qwen/Qwen3-8B", QWEN_MODEL)
clone_if_missing("https://huggingface.co/nomic-ai/nomic-embed-text-v1.5", EMBED_MODEL)

# ===============================
# List all files in models
# ===============================
def list_model_files(model_path: Path):
    print(f"\nListing files under {model_path}:")
    if not model_path.exists():
        print(f"  [ERROR] Path does not exist: {model_path}")
        return
    for path in model_path.rglob("*"):
        if path.is_file():
            print(f"  {path.relative_to(HOME_DIR)} (size={path.stat().st_size} bytes)")

list_model_files(QWEN_MODEL)
list_model_files(EMBED_MODEL)

# ===============================
# Start the main Python app
# ===============================
print(f"\nStarting ECS AI Ops from {APP_DIR}/app.py on port 8080...")

# Build the command
cmd = [
    "python",
    str(APP_DIR / "app.py"),
    "--host", "0.0.0.0",   # listen on all interfaces
    "--port", "8080",
    "--model-dir", str(QWEN_MODEL),
    "--embed-dir", str(EMBED_MODEL),
]

# Redirect output to log file
with open(LOG_FILE, "w") as f:
    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

print(f"Application started, logging to {LOG_FILE}")

# Wait for the app to finish (keeps container running)
process.wait()