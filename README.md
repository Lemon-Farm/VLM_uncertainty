# VLM Uncertainty

Research code for BLIP-family vision-language model inference, vision embeddings, and uncertainty experiments.

## Current Pipeline

```text
Hugging Face dataset
  -> data/interim/imagenet/<split>
  -> DataLoader applies BLIP processor on the fly
  -> BLIPWrapper inference
  -> outputs/*.jsonl
```

The project no longer saves BLIP-ready `.pt` tensors as an intermediate dataset. It stores only the raw Hugging Face dataset with `save_to_disk`, then applies the BLIP processor at training or inference time.

## Directory Tree

```text
VLM_uncertainty/
+-- configs/                    # Dataset/model/metric/uncertainty config
+-- data/
|   +-- raw/                    # Optional manually managed raw files
|   +-- interim/                # Saved Hugging Face datasets
+-- outputs/                    # Inference and experiment outputs
+-- reports/                    # Analysis outputs and figures
+-- scripts/                    # CLI entry points
+-- src/vlm_uncertainty/
|   +-- data/                   # Dataset loading and BLIP processor wrappers
|   +-- evaluation/             # Inference loops and prediction export
|   +-- metrics/                # Evaluation and calibration metrics
|   +-- models/                 # BLIPWrapper
|   +-- uncertainty/            # Uncertainty scoring methods
+-- tests/                      # Tests
```

## Usage

Download the dataset:

```bash
python scripts/prepare_imagenet.py --split validation --max-samples 1000
```

Run BLIP inference with on-the-fly preprocessing:

```bash
python scripts/run_blip_inference.py \
  --dataset-dir data/interim/imagenet/validation \
  --output outputs/captions.jsonl
```
