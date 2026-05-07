# SPOHE-VGGNet: Segmentation using Particle Swarm Optimized Histogram Equalization and a Modified U-Net with VGG-19 for Brain Tumor Detection

## Overview

SPOHE-VGGNet is a deep learning framework for brain tumor MRI segmentation that integrates Particle Swarm Optimized Histogram Equalization (PSO-HE) with a modified U-Net architecture using a VGG19 encoder backbone. The framework is designed to improve tumor boundary visibility, enhance feature extraction, and increase segmentation robustness across heterogeneous MRI modalities.

The proposed approach combines adaptive preprocessing and convolutional feature learning to achieve accurate segmentation performance while maintaining computational practicality for GPU-based medical imaging workflows.

---

## Preprint Publication

The initial version of this work is available as a Research Square preprint:

Saifullah, S., & Dreżewski, R.
**SPOHE-VGGNet: Segmentation using Particle Swarm Optimized Histogram Equalization and a Modified U-Net with VGG-19 for Brain Tumor Detection**.
Research Square Preprint, Version 1, 23 October 2025.

DOI: https://doi.org/10.21203/rs.3.rs-6870596/v1

---

## Key Contributions

* PSO-HE adaptive preprocessing for MRI contrast enhancement
* Modified U-Net architecture with VGG19 encoder
* Enhanced tumor-boundary delineation
* Grad-CAM interpretability analysis
* Error-map visualization
* Five-fold cross-validation evaluation
* Statistical robustness analysis using Wilcoxon signed-rank testing
* Computational profiling and inference-time analysis

---

## Architecture

The proposed SPOHE-VGGNet framework consists of:

1. PSO-HE preprocessing stage
2. VGG19 encoder
3. U-Net decoder with skip connections
4. Segmentation prediction layer
5. Grad-CAM interpretability module

---

## PSO-HE Preprocessing

The PSO-HE preprocessing stage optimizes gamma-based histogram equalization using Particle Swarm Optimization.

### Optimization Settings

| Parameter          | Value                |
| ------------------ | -------------------- |
| Swarm Size         | 30                   |
| Gamma Bounds       | 0.95 – 1.05          |
| Iterations         | 50–100               |
| Objective Function | Entropy Maximization |

The preprocessing stage improves:

* tumor visibility,
* local contrast,
* structural preservation,
* segmentation stability.

---

## Dataset

### Figshare Brain Tumor Dataset (FBTS)

Tumor classes:

* Meningioma
* Glioma
* Pituitary

### BraTS 2021 Dataset

MRI modalities:

* FLAIR
* T1
* T1CE
* T2

Input resolution:

* 256 × 256 × 3

---

## Experimental Setup

| Component     | Configuration         |
| ------------- | --------------------- |
| Framework     | TensorFlow / Keras    |
| GPU           | NVIDIA A100-SXM4-40GB |
| Batch Size    | 8                     |
| Epochs        | 50                    |
| Optimizer     | Adam                  |
| Learning Rate | 1e-3                  |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/SPOHE-VGGNet-BrainTumorSegmentation.git

cd SPOHE-VGGNet-BrainTumorSegmentation

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Testing

```bash
python test.py
```

---

## Grad-CAM Visualization

The repository includes Grad-CAM-based interpretability analysis for evaluating spatial activation behavior and tumor-focused feature learning.

Generated outputs include:

* Grad-CAM overlays
* Error maps
* Predicted masks
* Ground-truth comparisons

---

## Statistical Analysis

The repository includes:

* Wilcoxon signed-rank testing
* Cross-validation evaluation
* Mean ± standard deviation analysis
* Confidence interval computation

Example:

```bash
python evaluation/statistical_analysis.py
```

---

## Computational Profiling

| Metric                 | Value     |
| ---------------------- | --------- |
| Parameters             | 31.172 M  |
| Training Time / Epoch  | 7.471 s   |
| Total Training Time    | 374.25 s  |
| Inference Time / Image | 8.97 ms   |
| PSO-HE Time / Image    | 22.603 ms |

---

## Repository Structure

```text
SPOHE-VGGNet/
│
├── datasets/
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── visualization/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{saifullah2025spohevggnet,
  title = {SPOHE-VGGNet: Segmentation using Particle Swarm Optimized Histogram Equalization and a Modified U-Net with VGG-19 for Brain Tumor Detection},
  author = {Shoffan Saifullah and Rafał Dreżewski},
  journal = {Research Square},
  year = {2025},
  doi = {10.21203/rs.3.rs-6870596/v1},
  url = {https://doi.org/10.21203/rs.3.rs-6870596/v1}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This work was supported by GPU computational resources using NVIDIA A100-SXM4-40GB infrastructure.
