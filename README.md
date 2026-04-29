# 3D Liver Tumour Segmentation with U-Net

This project implements a 3D U-Net for liver and tumour segmentation on CT volumes from the LiTS (Liver Tumour Segmentation) challenge dataset. It was built as part of my preparation for doctoral research in medical image analysis, specifically to develop practical experience with volumetric deep learning pipelines before applying to the PhD position on personalised radiotheranostics at the Université libre de Bruxelles.

The work covers the full pipeline from raw NIfTI data loading and CT windowing through model training, validation with Dice metrics, and prediction visualisation — implemented from scratch using PyTorch and MONAI on Kaggle's free GPU tier.
<img width="780" height="511" alt="Screenshot 2026-04-29 at 2 13 01 PM" src="https://github.com/user-attachments/assets/2ef0843d-2c27-42f1-a46c-f2feef001432" />

---

## Motivation

Radiotheranostics is an emerging approach in nuclear medicine that combines targeted molecular imaging with radionuclide therapy to enable patient-specific diagnosis and treatment. One of the core challenges in this space is building robust pipelines that can integrate heterogeneous imaging data — PET/SPECT, CT, MRI — and produce reliable anatomical segmentations that feed downstream patient stratification models.

This project is a deliberate first step into that domain. Liver and tumour segmentation on CT is a well-studied benchmark that shares the fundamental difficulties of clinical imaging work: severe class imbalance, variable volume geometry, inconsistent voxel spacing across scanners, and the need to reason in three dimensions rather than treating each slice independently. Working through these problems concretely was more valuable to me than reading about them.

---

## Dataset

The LiTS dataset (Liver Tumour Segmentation Challenge) consists of 131 abdominal CT volumes with pixel-level annotations. Each volume is labelled at the voxel level with three classes: background (0), liver (1), and tumour (2). The dataset is publicly available on Kaggle via the `andrewmvd/liver-tumor-segmentation` repository.

A key characteristic of this dataset is extreme class imbalance. In a typical volume, background voxels account for roughly 97% of the total, liver for around 2.8%, and tumour for under 0.01%. This distribution directly motivates the choice of DiceCE loss over standard cross-entropy, and it means that tumour Dice is a genuinely hard metric to move — a model can achieve high accuracy by predicting no tumour at all.

Due to Kaggle dataset availability, 51 volumes had matched image-segmentation pairs and were used for this experiment. The split was 40 volumes for training and 11 for validation.

---

## Architecture

The model is a 3D U-Net implemented via MONAI's `UNet` class with the following configuration:

```
spatial_dims  = 3
in_channels   = 1       (CT is single-channel / grayscale)
out_channels  = 3       (background, liver, tumour)
channels      = (16, 32, 64, 128, 256)
strides       = (2, 2, 2, 2)
num_res_units = 2
dropout       = 0.1

Total parameters: 4,807,482
Model size: ~19.2 MB (float32)
```

The choice of 3D convolutions rather than 2D slice-by-slice processing is deliberate. Liver and tumour boundaries are three-dimensional structures — processing axial slices independently discards the contextual information in adjacent slices that makes segmentation tractable. The skip connections between encoder and decoder levels preserve the fine spatial detail that gets compressed during downsampling, which matters especially for small tumour regions.

---

## Preprocessing

CT windowing is applied before any model input. Raw Hounsfield Unit values in this dataset range from roughly -3024 to +1410 HU. Without windowing, the liver occupies a narrow band in that range and is essentially invisible when normalised naively. The standard soft tissue window (WL = 50, WW = 400) clips values to the -150 to +250 HU range and normalises to [0, 1], which is what the transforms pipeline applies via `ScaleIntensityRanged`.

The training transforms chain is: load NIfTI, ensure channel-first layout, apply intensity windowing, crop foreground to remove the large air border around the patient body, then randomly sample two patches of size 128 × 128 × 64 per volume with balanced positive and negative sampling (`RandCropByPosNegLabeld`, pos=1, neg=1). Random flips along all three axes and 90-degree rotations are applied for augmentation. Validation uses the same windowing and foreground crop but evaluates on full volumes via sliding window inference with 25% overlap.

---

## Training

```
Loss:       DiceCELoss (Dice + Cross Entropy combined)
Optimizer:  AdamW (lr = 1e-4, weight_decay = 1e-5)
Scheduler:  ReduceLROnPlateau (mode = max, factor = 0.5, patience = 5)
Metric:     DiceMetric (background excluded)
Hardware:   Tesla T4 GPU, Kaggle free tier
```

Training ran for 50 epochs, stopping due to Kaggle session time constraints (~5.5 hours total). The best checkpoint was saved at epoch 46.

| Metric | Value |
|---|---|
| Best Liver Dice | 0.4536 |
| Best Tumour Dice | 0.0000 |
| Best Mean Dice | 0.4536 |
| Best epoch | 46 |
| Training time | ~5.5 hours |

Tumour Dice remained at 0.0 throughout all 50 epochs. This is an expected outcome given the combination of factors: the dataset used here contains 51 volumes rather than the full 131, training was capped at 50 epochs, and tumour regions represent a fraction of a percent of total voxels in most cases. In practice, tumour Dice typically begins to improve only after the model has converged on liver segmentation — which was still improving at epoch 50. Extended training on the full dataset would be the straightforward next step.

The training loss dropped steadily from 1.97 at epoch 1 to 0.82 at epoch 50, confirming that the model was learning throughout the run.

<img width="781" height="279" alt="Screenshot 2026-04-29 at 2 12 44 PM" src="https://github.com/user-attachments/assets/ac4476ad-f1fe-458b-b6fd-a2509a009221" />


---

## Limitations and next steps

The most important limitation is compute. Running on Kaggle's free tier with a session time limit meant training on a subset of the data for a bounded number of epochs. A natural continuation would be to train on all 131 volumes for 150 to 200 epochs, which would likely push liver Dice above 0.85 and allow tumour segmentation to emerge. Adding class-weighted sampling or focal loss modifications to the DiceCE objective could also help with the tumour imbalance.

A second direction is connecting this work to the actual modality used in radiotheranostics. The pipeline built here generalises directly to PET/SPECT volumes, which share the NIfTI format and similar preprocessing considerations. Adapting the transforms for SUV normalisation instead of HU windowing, and extending the label space to include additional structures relevant to PSMA PET analysis, would be the natural bridge between this project and the clinical application.

---

## Repository structure

```
notebooks/
    day1-data-exploration.ipynb     — data loading, visualisation, intensity analysis
    day2-5-unet-training.ipynb      — full pipeline: transforms, model, training, results
results/
    slice_visualization.png         — axial, coronal, sagittal views with mask overlay
    intensity_histogram.png         — HU distribution before and after windowing
    training_curves.png             — loss and Dice over 50 epochs
requirements.txt
README.md
```

---

## How to run

The notebooks are designed to run on Kaggle. Add the `andrewmvd/liver-tumor-segmentation` dataset to your notebook via the Data panel, then run cells sequentially. GPU acceleration is required for the training notebook.

```bash
pip install monai nibabel matplotlib torch torchvision
```

The full training notebook is also available as a committed Kaggle version and can be forked directly.

---

## Environment

```
Python       3.12
PyTorch      2.10.0+cu128
MONAI        1.4.0
CUDA         12.8
GPU          Tesla T4 (Kaggle)
```
