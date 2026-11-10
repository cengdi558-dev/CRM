# CRM — Concomitance Relationship Mining with Disparity Synapse Modeling for Facial Expression Recognition

Reference implementation of **CRM**, a facial-expression-recognition model that mines
inter-AU *concomitance* relationships and models them as a directed **disparity synapse**
graph. The pipeline is

```
image ──► COC (occlusion compensation) ──► SSA (Sinkhorn token alignment)
      ──► STM (top-κ tokenisation)     ──► SIR (directed W = S + A synapse graph)
      ──► classifier
```

* **COC** — predicts eight AU responses, estimates an occlusion map and compensates
  the hidden responses with a symmetry prior.
* **SSA** — aligns visual tokens with the semantic profiles of the eight AU regions by
  entropic optimal transport (Sinkhorn).
* **STM** — keeps the κ most informative tokens plus one token per AU region.
* **SIR** — a stack of layers whose adjacency is decomposed into a symmetric
  concomitance matrix `S` and an antisymmetric disparity matrix `A`; `S` is initialised
  from the data-driven co-occurrence prior `C`.

## Results (RAF-DB, our run)

| Model | Val accuracy |
|---|---|
| ResNet-50 | 77.1% |
| ViT-B/16 | 74.5% |
| EfficientNet-B0 | 75.7% |
| ConvNeXt-T | 78.4% |
| Swin-T | 86.1% |
| **CRM (Swin-B/384, 24 epochs)** | **87.9%** |

## Installation

```bash
python -m pip install -r requirements.txt
```

`torch >= 2.0` and `timm >= 1.0` are required. The code picks `torch.cuda.amp` or
`torch.amp` automatically, so it runs on both older (2.0–2.3) and newer (2.4+) stacks.

## Data

RAF-DB is downloaded from a public mirror and converted to an `ImageFolder` layout:

```bash
python -m crm.fetch_rafdb --out .rafdb
```

This produces `.rafdb/{train,val,test}/<class>/*.jpg` (12 271 / 1 225 / 3 068 images).

## Training

```bash
# 1. data-driven co-occurrence prior C (8x8)
python -m crm.estimate_c --data .rafdb/train --out c.npy

# 2. cached AU-region semantics for every image
python -m crm.gen_semantics --data .rafdb --out sem.npz --dim 1024

# 3. train
python -m crm.train --data .rafdb --prior c.npy --semantics sem.npz \
    --backbone swin_base_patch4_window12_384 --image-size 384 \
    --batch-size 4 --accum 16 --epochs 24 --out crm_best.pth
```

> **AMP note.** On some GPUs mixed precision produced non-finite gradients in the
> Sinkhorn branch. The training loop skips non-finite batches and sanitises gradients,
> but if you see `nan` losses, set `AMP=0` (pure fp32). All reported numbers were
> obtained with `AMP=0`.

## Figures

```bash
# structural figures (regions, prior, occlusion, synapse, qualitative, confusion, curves)
python -m crm.figures --data .rafdb --ckpt checkpoints/crm_best.pth --log train.log \
    --out figs --which all

# qualitative heatmap comparison (Fig. 9): five baselines + CRM
BASELINE_DIR=checkpoints/compare CRM_CKPT=checkpoints/crm_best.pth \
    python -m scripts.fig9_heatmaps

# robustness under occlusion / low resolution / blur / rotation / low light (Fig. 10)
BASELINE_DIR=checkpoints/compare CRM_CKPT=checkpoints/crm_best.pth \
    CRM_NOCOC_CKPT=checkpoints/crm_nococ.pth python -m scripts.fig10_robustness
```

`checkpoints/compare/<model>.pth` are baseline checkpoints produced by

```bash
python -m crm.compare --data .rafdb --out checkpoints/compare \
    --models resnet50,vit_base_patch16_224,swin_tiny_patch4_window7_224,efficientnet_b0,convnext_tiny \
    --image-size 224 --epochs 8 --samples 10
```

## Ablation

Drop the occlusion compensation to reproduce the `CRM w/o COR` column:

```bash
python -m crm.train ... --no-coc --out crm_nococ.pth
```

## Repository layout

```
crm/
  model.py        CRM assembly (COC → SSA → STM → SIR)
  coc.py          occlusion compensation
  ssa.py          Sinkhorn token alignment
  stm.py          top-κ tokenisation
  sir.py          directed synapse graph layers
  losses.py       classification + alignment + budget + sparsity terms
  regions.py      eight AU regions and their masks
  train.py        training / evaluation loop
  figures.py      structural figures
  compare.py      baseline trainer for the qualitative comparison
  fetch_rafdb.py  dataset downloader
scripts/
  fig9_heatmaps.py      Fig. 9
  fig10_robustness.py   Fig. 10
  docx_replace.py       swap a figure inside a .docx
```

## License

Released for academic use. Please cite the paper if you use this code.
