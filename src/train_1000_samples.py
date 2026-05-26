"""
One-time script to train HazardVLM on a random subset of 1000 samples.

Run from the src/ directory:
    python train_1000_samples.py

What it does:
1. Loads the full dataset JSON specified in config/model_config_vlm.yaml
2. Randomly samples 1000 entries (seed=42 for reproducibility)
3. Writes the subset to a temporary JSON file
4. Temporarily updates the config to point at the subset
5. Runs the standard training pipeline (main.py)
6. Restores the original config and cleans up temp files
"""

import json
import os
import random
import runpy
import shutil
import sys

import yaml


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 1. Load original config to find the dataset filename
# ---------------------------------------------------------------------------
CONFIG_PATH = "config/model_config_vlm.yaml"
CONFIG_BACKUP_PATH = "config/model_config_vlm.yaml.bak"

with open(CONFIG_PATH, encoding="utf-8") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

orig_filename = model_config["input_filename"]
orig_json_path = f"datasets/{orig_filename}.json"

# ---------------------------------------------------------------------------
# 2. Build 1000-sample subset
# ---------------------------------------------------------------------------
with open(orig_json_path, encoding="utf-8") as f:
    full_data = json.load(f)

total = len(full_data)
if total < 1000:
    raise ValueError(
        f"Dataset only has {total} samples; cannot extract 1000."
    )

random.seed(42)
subset = random.sample(full_data, 1000)

subset_filename = f"{orig_filename}_1000"
subset_json_path = f"datasets/{subset_filename}.json"

with open(subset_json_path, "w", encoding="utf-8") as f:
    json.dump(subset, f)

print(f"[train_1000_samples] Created subset: {subset_json_path}")
print(f"[train_1000_samples] Full dataset: {total} -> Subset: 1000")

# ---------------------------------------------------------------------------
# 3. Temporarily swap config to point at subset
# ---------------------------------------------------------------------------
shutil.copy(CONFIG_PATH, CONFIG_BACKUP_PATH)

with open(CONFIG_PATH, encoding="utf-8") as f:
    lines = f.readlines()

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    for line in lines:
        if line.strip().startswith("input_filename:"):
            f.write(f"input_filename: '{subset_filename}'\n")
        else:
            f.write(line)

print(f"[train_1000_samples] Temporarily updated {CONFIG_PATH}")

# Verify the config file was actually modified
with open(CONFIG_PATH, encoding="utf-8") as f:
    verify_config = yaml.load(f, Loader=yaml.FullLoader)
print(f"[train_1000_samples] Verifying config input_filename: {verify_config['input_filename']}")

# ---------------------------------------------------------------------------
# 4. Run standard training pipeline in-process (avoids uv-run fs isolation)
# ---------------------------------------------------------------------------
try:
    print("[train_1000_samples] Launching main.py in-process ...\n")
    runpy.run_path("main.py", run_name="__main__")
finally:
    # -----------------------------------------------------------------------
    # 5. Restore original config and remove temp files
    # -----------------------------------------------------------------------
    shutil.move(CONFIG_BACKUP_PATH, CONFIG_PATH)
    os.remove(subset_json_path)
    print(f"\n[train_1000_samples] Restored {CONFIG_PATH}")
    print(f"[train_1000_samples] Removed temp dataset: {subset_json_path}")
