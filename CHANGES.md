# Confidence-Aware Memory Filtering — Code Changes

## Overview

Added confidence-aware memory gating to MemSAM to mitigate error propagation in video segmentation. Low-confidence predictions are down-weighted before writing to memory, preventing unreliable frames from corrupting subsequent predictions.

## Modified Files

### 1. `models/segment_anything_memsam/modeling/memsam.py`

**Change**: Added soft confidence gate in `_forward_with_memory()`.

**What was added**:
- Two new attributes in `__init__`: `confidence_threshold` (default 0.8) and `confidence_scale` (default 10.0)
- In the memory write block (line ~282): compute prediction entropy, derive frame-level confidence, apply sigmoid soft gate to memory values before storing

**Before** (baseline):
```python
v16, hidden = self.memory('encode_value', imgs[:,ti], imge[:,ti], hidden, mask, is_deep_update=is_deep_update)
values = torch.cat([values, v16], 3)
```

**After** (ours):
```python
# Confidence-aware memory gating
prob = torch.sigmoid(mask)
entropy = -(prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
frame_confidence = 1.0 - entropy.mean()
gate = torch.sigmoid(self.confidence_scale * (frame_confidence - self.confidence_threshold))

v16, hidden = self.memory('encode_value', imgs[:,ti], imge[:,ti], hidden, mask, is_deep_update=is_deep_update)
v16 = v16 * gate  # scale memory value by prediction confidence
values = torch.cat([values, v16], 3)
```

### 2. `train_video.py`

**Change**: Added `--confidence_threshold` argument (default 0.8) and pass it to model after construction.

## How It Works

1. After each frame's mask prediction, compute pixel-wise binary entropy
2. Average entropy across all pixels → frame-level uncertainty
3. Frame confidence = 1 - mean_entropy
4. Soft gate = sigmoid(scale × (confidence - threshold))
5. Multiply memory value by gate before storing → low-confidence frames contribute less to memory

## Usage

```bash
# Baseline (original MemSAM)
python train_video.py --modelname MemSAM --task CAMUS_Video_Full --enable_memory --reinforce

# Ours (with confidence-aware memory filtering, default threshold=0.8)
python train_video.py --modelname MemSAM --task CAMUS_Video_Full --enable_memory --reinforce --confidence_threshold 0.8

# Ablation: different thresholds
python train_video.py --modelname MemSAM --task CAMUS_Video_Full --enable_memory --reinforce --confidence_threshold 0.7
```

## Results Comparison

| Method | DSC (%) | HD (mm) | Notes |
|--------|---------|---------|-------|
| MemSAM (baseline) | — | — | Original, no filtering |
| Ours (τ=0.7) | — | — | |
| Ours (τ=0.8) | — | — | Recommended |
| Ours (τ=0.9) | — | — | |

*Fill with real numbers after training.*
