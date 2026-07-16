# MemSAM + CAMG (Ours)
[**MemSAM: Taming Segment Anything Model for Echocardiography Video Segmentation**](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_MemSAM_Taming_Segment_Anything_Model_for_Echocardiography_Video_Segmentation_CVPR_2024_paper.pdf), CVPR 2024, _Oral_

This repository contains the official MemSAM codebase extended with our novel **Confidence-Aware Memory Gating (CAMG)** mechanism. CAMG dynamically scales memory update values based on prediction confidence to suppress error propagation and acoustic-speckle noise contamination in echocardiography videos.

<div align=center>
<img width="753" height="498" alt="image" src="https://github.com/user-attachments/assets/069420a2-4386-47d3-b2f8-f5190b9eb335" />
</div>


## Key Extensions (CAMG)
- **Dynamic Memory Gating:** Scales memory writes using a soft sigmoid gate derived from the model's predicted confidence maps.
- **VRAM Optimizations:** Sequential frame-by-frame encoding inside `encode_key` reducing peak training GPU memory by over 90% (making ViT-B training comfortable on 8GB GPUs).
- **AMP Support:** Out-of-the-box Automatic Mixed Precision (`torch.cuda.amp`) integration.

## Installation
```bash
conda create --name memsam python=3.10
conda activate memsam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge scikit-image scipy pandas seaborn easydict einops batchgenerators medpy tensorboard -y
```

## Usage
### 1. Dataset Preprocessing
Download CAMUS dataset, then run:
```bash
python utils/preprocess_camus.py -i CAMUS_public/database_nifti -o dataset/SAMUS/CAMUS_full -f CAMUS_public/camus_split.json
```

### 2. Pretrained Weights
Download the [ViT-B SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and save it to `checkpoints/sam_vit_b_01ec64.pth`.

### 3. Training
- **Train our CAMG model (Recommended, VRAM optimized):**
  ```bash
  python train_video.py --modelname MemSAM --task CAMUS_Video_Full --enable_memory --reinforce --confidence_threshold 0.8 --batch_size 1 --frame_length 3 --keep_log --warmup
  ```
- **Train the original MemSAM baseline:**
  ```bash
  python train_video.py --modelname MemSAM --task CAMUS_Video_Full --enable_memory --reinforce --disable_confidence_gating --batch_size 1 --frame_length 3 --keep_log --warmup
  ```

### 4. Evaluation
Evaluate checkpoints using:
```bash
python test_video.py
```
*(Toggle baseline vs CAMG options inside `test_video.py` EasyDict args).*

## Results & Project Report
Please refer to [REPORT.md](./REPORT.md) for our full experimental analysis, ablation studies, clinical ejection fraction (EF) analysis, and qualitative evaluation.

## Acknowledgement
The work is based on [SAM](https://github.com/facebookresearch/segment-anything), [SAMUS](https://github.com/xianlin7/SAMUS) and [XMem](https://github.com/hkchengrex/XMem). Thanks for the open source contributions to these efforts!

## Citation
if you find our work useful, please cite our paper, thank you!
```
@InProceedings{Deng_2024_CVPR,
    author    = {Deng, Xiaolong and Wu, Huisi and Zeng, Runhao and Qin, Jing},
    title     = {MemSAM: Taming Segment Anything Model for Echocardiography Video Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9622-9631}
}
```
