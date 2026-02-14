# PDC-KD: Physics-Guided Dual-Consistency Knowledge Distillation

> **Official Preview Implementation** for the paper:
> *Parameter Manifold Alignment and Physics-Guided Learning for Heterogeneous Nonlinear Dynamics Estimation in Wearable Systems*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

---

## ⚠️ Important Notice: Code Redaction for Blind Review

**This repository contains a preview version of the PDC-KD framework.**

To uphold the integrity of the **double-blind review process** and protect core intellectual property prior to formal publication, specific algorithmic implementations have been redacted or simplified in this release:

1.  **Physics-Guided Constraints:** The Cholesky-based parameterization of the equivalent inertia matrix and the explicit Newton–Euler dynamics formulation are replaced with placeholders.
2.  **Manifold Alignment:** The Fisher Information Matrix (FIM) calculation and SVD-based subspace projection logic are simplified to standard MSE in this preview.
3.  **Data:** The script uses `DemoDataset` (synthetic data) to demonstrate the pipeline without requiring access to the private clinical dataset.

> **Note:** The full, unredacted codebase will be released upon the paper's acceptance.

---

## Abstract

Real-time monitoring of human joint dynamics on edge devices is constrained by limited computing resources. Lightweight deployment often necessitates removing computationally intensive time–frequency preprocessing (e.g., CWT), resulting in a regression from 2D representations to 1D time-series signals.

To address the resulting observability loss, we propose **PDC-KD (Physically Guided Dual-Consistency Knowledge Distillation)**. This framework introduces:

1.  **Parameter-Manifold Alignment:** A shared anchor space aligning teacher and student parameter geometry using Fisher information.
2.  **Physics-Guided Compensation:** An exogenous mechanism incorporating biomechanical priors to correct non-physical predictions.

---

## Repository Structure

```text
PDC-KD-Preview/
├── main.py          # Self-contained training & architecture demo
├── README.md        # Project documentation
├── requirements.txt # Dependencies
└── LICENSE          # MIT License


## Getting Started

### Prerequisites

Before running the demo, ensure the following dependencies are installed:

-Python 3.8+

-PyTorch 2.0+

-NumPy

Check your Python version:

```bash
python --version
```


### Installation

Clone the repository and install required packages:

```bash
git clone [https://github.com/Anonymous-Repo/PDC-KD.git](https://github.com/Anonymous-Repo/PDC-KD.git)
cd PDC-KD
pip install -r requirements.txt
```

It is recommended to use a virtual environment (e.g., `venv` or `conda`).

---

### Running the Demo

The preview script is fully self-contained and automatically generates synthetic data to verify the architecture and training flow.

Run:

```bash
python main.py
```

---

### Expected Output

If execution is successful, you should see output similar to:

```Plaintext
[Start] Training Loop (Preview Version)
Running PDC-KD (Preview Mode) on cuda...
Epoch [1/2] | Loss: 0.8234 (Physics/Struct details hidden)
Epoch [2/2] | Loss: 0.6120 (Physics/Struct details hidden)
[Done] Execution successful.
```


---

### Notes

* This is a **preview version** for double-blind review.
* Physics-guided constraints and manifold alignment internals are redacted.
* Synthetic data is used to demonstrate pipeline functionality.

---

## Model Architecture (Simplified)

The preview implementation demonstrates the structural design of the PDC-KD framework while omitting proprietary algorithmic details. The framework consists of two networks trained under a dual-consistency distillation objective.

### Teacher Model

**Input:**
 2D Continuous Wavelet Transform (CWT) spectrograms [B, 32, 115, T]

**Backbone:**
 CNN encoder + Bidirectional LSTM

**Role:**
Extracts high-fidelity representations to serve as supervisory reference.

---

### Student Model

**Input:**
1D Raw IMU time-series signals `[B, T, 32]`

**Backbone:**
Lightweight GRU + Linear adaptation head

**Role:**
Designed for real-time edge inference (  ms latency)

---

### Distillation Strategy (Preview Version)

The training pipeline integrates three conceptual components:

1. **Output Consistency**

   Standard regression-based supervision between teacher and student outputs.
2. **Parameter Manifold Alignment**

   In the full version, Fisher Information-based alignment constrains parameter geometry.

   In this preview, it is simplified to standard MSE alignment.
3. **Physics-Guided Compensation**

   Newton–Euler dynamics-based constraints are conceptually included,

   but core formulations are redacted in this release.

> The preview focuses on architectural transparency rather than full methodological disclosure.
>

---

## Data Source

The experiments in this study were conducted using a publicly available dataset. We extend our gratitude to the authors for making their data accessible.

- **Dataset Title:** "A Human Lower-Limb Biomechanics and Wearable Sensors Dataset During Cyclic and Non-Cyclic Activities"
- **Authors:** Scherpereel, K., Molinaro, D., Inan, O. et al.
- **Availability:** The dataset is available at the [**Georgia Tech Repository**](https://repository.gatech.edu/entities/publication/20860ffb-71fd-4049-a033-cd0ff308339e/).
- **License:** The dataset is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. This means you are free to share and adapt the data, provided you give appropriate credit to the original authors.

---

## License

The source code in this repository is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for more details.

---

## Citation

If you find our work or this repository useful in your research, please consider citing our paper:

```
@article{PDC_KD_2026,
  title={Parameter Manifold Alignment and Physics-Guided Learning for Heterogeneous Nonlinear Dynamics Estimation in Wearable Systems},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

## Email:

If you have any questions, please email to: shuxu@mail.ustc.edu.cn
