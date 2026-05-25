# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HazardVLM is a PyTorch-based video language model that generates natural language hazard warnings from driving video. It uses an X3D video encoder (from `pytorchvideo`) to extract spatiotemporal features from video frames, passes them through a visual MLP, and uses a Transformer decoder to autoregressively generate hazard captions (e.g., "Hazardous lane exit by car on right").

## Common Commands

All execution happens from the `src/` directory:

```bash
cd src

# Training + evaluation (default)
python main.py

# Model visualization (Eigen-CAM heatmaps)
python model_visual.py
```

There is no test suite, linter, or build step configured. Code quality is manual.

## Configuration Architecture

Two YAML files control behavior:

**`src/config/init_config.yaml`** — Runtime mode and environment:
- `mode`: `'vlm'` or `'benchmark'`
- `proc_mode`: `1` (train+eval), `2` (eval pretrained), `3` (transfer learning)
- `single_run_bool`: `1` for single run, `0` for W&B sweep
- `wandb_bool`: `1` to log to W&B (requires `wandb_account_name`), `0` for offline/debug
- `compile`: `1` to enable `torch.compile()` (Linux only)
- `target_gpu`: e.g. `'cuda:0'`

**`src/config/model_config_vlm.yaml`** — Model hyperparameters and dataset:
- `input_filename`: Dataset JSON stem (expects `src/datasets/{input_filename}.json`)
- `encoder`: `'x3d_m'`, `'x3d_l'`, or `'pre_extracted'`
- `model_load_path`: Pretrained model filename (no extension) for proc_mode 2/3
- `batch_size` / `batch_multiplier`: Effective batch = batch_size * batch_multiplier (gradient accumulation)
- `visual_mlp_output_dim`: Must be divisible by `decoder_heads`
- `model_visualiser_mode`: Must be `True` for visualization script

**`src/config/model_config_vlm_sweep.py`** — W&B sweep parameter grid (used when `single_run_bool: 0`).

## Key Source Architecture

**`src/main.py`** — Entry point. Loads both config files, initializes `HazardVLM`, datasets, dataloaders, optimizer, and calls `train_model()` then `eval_model()` twice (full video + video with end frames removed).

**`src/modules/processing_layers/model_vlm.py`** — `HazardVLM` class:
- `encoder`: X3D feature extractor via `torchvision.models.feature_extraction.create_feature_extractor`
- `visual_mlp`: Optional MLP projecting encoder output (9408-dim) to decoder input dim
- `transformer_decoder`: Standard PyTorch `nn.TransformerDecoder` with learned positional embeddings
- `forward()`: Training uses teacher forcing (stochastic pattern per batch); inference generates greedily. Returns `(logits, activations)` where activations are used by the visualizer.

**`src/modules/training_loop/train_eval_vlm.py`**:
- `train_model()`: Full training loop with pseudo-batch gradient accumulation, mixed precision (`GradScaler`), checkpointing per epoch (`*_checkpoint.pt`), early stopping on validation loss plateau, and teacher forcing decay.
- `eval_model()`: Inference on test set. Computes BLEU/ROUGE/etc. via `pycocoevalcap`. Also extracts hazard/actor/location components from captions for per-component metrics.

**`src/modules/input_layers/dataloader.py`** — `HazardVideoDataset`:
- Loads videos via OpenCV (`cv2.VideoCapture`)
- Frames normalized to `[0, 1]`
- Supports drawing hazardous actor bounding boxes (`show_haz_actor_bbox`)
- Supports runtime adaptive sampling (`adaptive_frame_sample: True`) or loading pre-sampled datasets
- `collate_fn` pads videos to max length in batch and returns `(videos, masks, captions)`

**`src/modules/input_layers/adaptive_sampling_tools.py`** — Optical flow-based adaptive frame sampling:
- Uses Farneback optical flow (CPU or CUDA) to compute motion magnitude between frames
- Three modes: `uniform` (highest motion per chunk), `highest_value` (global top-N), `random`
- Threading variant (`get_sample_frames_threading`) for uniform mode
- Note: For training speed, datasets are typically preprocessed with adaptive sampling rather than using runtime sampling

**`src/modules/model_visualiser/model_visualiser.py`** — `VisualiseActivations`:
- Loads trained model and runs inference on videos in `src/modules/model_visualiser/video/`
- Extracts intermediate encoder activations (defined in `dict_extra_layers` inside `model_vlm.py`)
- Generates Eigen-CAM 2D projections of activations per timestep
- Outputs to `src/modules/model_visualiser/output/{model_name}/`

## Dataset and Model Files

Datasets and pretrained models are not in the repo. Download links are in the README.

- **Models** go in `src/models/` (e.g., `HazardVLM_69613_A.pt`)
- **Datasets** go in `src/datasets/` (JSON metadata + frame directories)

Dataset JSON is a list of scene dicts:
```python
{'filename': str, 'metadata': {
    'class_label': int, 'caption': str,
    'event_type': str, 'actor': str, 'location': str,
    'ego_involve': bool, 'night': bool,
    'anomaly_start': int, 'anomaly_end': int,
    'haz_actor_bbox': list  # per-frame bbox annotations
}}
```

## Important Conventions

- Always run from `src/`. All paths in code are relative to `src/`.
- Tokenizer files live in `src/config/tokenizer/` (HuggingFace `PreTrainedTokenizerFast` format).
- Model saves are named: `models/{model}_{dataset}_{random_id}_{repeat_run_id}.pt`
- Checkpoints are `{model_save_name}_checkpoint.pt` and are deleted after successful training.
- The visualizer script (`model_visual.py`) is standalone and runs at module level (not guarded by `if __name__ == '__main__'`).
- `restart_training: True` resumes from the checkpoint file matching `model_load_path`.
- Video resolution for training: 200x200 (matches DoTA-HEC dataset).
