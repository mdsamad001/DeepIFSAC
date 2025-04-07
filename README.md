# 1. DeepIFSAC

**DeepIFSAC** is a deep learning framework for tabular data that leverages attention-based architecture within a contrastive learning framework for missing value imputation. This repository provides code for data processing, training the DeepIFSAC model for missing value imputation on Tabular data set and a real-world EHR data set.

---

## 2. Table of Contents

1. [Overview](#3-overview)
2. [Installation](#4-installation)
3. [Dataset](#5-dataset)
4. [Training the Model](#6-training-the-model)
   - [Pretraining](#61-pretraining)
   - [Finetuning/Downstream Evaluation](#62-finetuningdownstream-evaluation)
5. [Results and Outputs](#7-results-and-outputs)
6. [Usage Examples](#8-usage-examples)
7. [Project Structure](#9-project-structure)

---

## 3. Overview

DeepIFSAC is designed to work with tabular datasets (e.g., from OpenML). It supports various tasks such as multiclass classification, binary classification, and regression. The model implements both a **pretraining phase** (with options for contrastive and denoising tasks) and a **finetuning phase** that trains downstream classifiers.

---

## 4. Installation

### 4.1 Clone the Repository

```bash
git clone https://github.com/yourusername/DeepIFSAC.git
cd DeepIFSAC
```

### 4.2 Create a Virtual Environment (optional but recommended)

If you use Conda, create an environment using the provided YAML file:

```bash
conda env create -f environment.yaml
conda activate DeepIFSAC
```

Ensure that you have PyTorch installed with the proper CUDA version if you intend to use a GPU. All dependencies and their versions are specified in the `environment.yaml` file.

---

## 5. Dataset

The code leverages a dataset loading function (`my_data_prep_openml`) located in the `data_openml` module. By providing a dataset ID (`--dset_id`), the code automatically downloads and processes the dataset from OpenML.

---

## 6. Training the Model

The training process is divided into two main phases: **pretraining** and **downstream finetuning**.

### 6.1 Pretraining

DeepIFSAC can be pretrained using various objectives (e.g., denoising, contrastive loss). To run pretraining, set the `--pretrain` flag and specify additional parameters (like number of pretrain epochs, augmentation type, missing rate, etc.). The pretraining function (`DeepIFSAC_pretrain`) takes care of data augmentation, computes losses over epochs, and saves training metrics.

### 6.2 Finetuning/Downstream Evaluation

After pretraining, the model can be finetuned on a downstream task. The repository supports:

- Training classical classifiers (e.g., Logistic Regression, Gradient Boosting) on features extracted from the model.
- Training a separate MLP head (using `simple_MLP`) for further finetuning.

---

## 7. Results and Outputs

### 7.1 Pretrained Model Weights

Model weights are saved under the `./results/model_weights` directory. The filename is generated based on parameters such as dataset ID, attention type, missing type, missing rate, dataset seed, and corruption type.

### 7.2 Training Metrics

Pretraining metrics (e.g., running loss per epoch) are saved as pickle files in the `./results/training_scores` directory.

### 7.3 Downstream Performance

After finetuning, the performance of both the classical classifiers (LR, GBT) and the MLP head is printed to the console.

---

## 8. Usage Examples

To train the DeepIFSAC model with pretraining for a multiclass task on dataset ID 11, run:

```bash
python my_train.py \
  --dset_id 11 \
  --task multiclass \
  --attentiontype colrow \
  --pretrain \
<<<<<<< HEAD
  --pretrain_epochs 1 \
  --epochs 2 \
=======
  --pretrain_epochs 1000 \
  --epochs 200 \
>>>>>>> a13f22d... DeepIFSAC pipeline [OpenML]
  --batchsize 128 \
  --dset_seed 0 \
  --cuda_device 0 \
  --use_default_model \
  --missing_rate 0.5 \
  --missing_type mcar
```

Adjust parameters as needed. Refer to the command-line argument help for more details:

```bash
python my_train.py --help
```

---

## 9. Project Structure

```
DeepIFSAC/
├── data_openml
│   ├── my_data_prep_openml.py      # Data processing and loading from OpenML.
│   └── ...
├── models
<<<<<<< HEAD
│   ├── deepifsac.py                # Implementation of the DeepIFSAC model.
│   ├── model.py                    # Additional models (e.g., simple_MLP).
│   └── ...
├── pretraining
│   └── DeepIFSAC_pretrain.py       # Pretraining functions for DeepIFSAC.
=======
│   ├── pretrainmodel.py                # Implementation of the DeepIFSAC model.
│   ├── model.py                    # Additional models (e.g., simple_MLP).
│   └── ...
>>>>>>> a13f22d... DeepIFSAC pipeline [OpenML]
├── utils
│   └── ...                         # Helper functions for training, evaluation, etc.
├── augmentations
│   └── ...                         # Data augmentation routines.
<<<<<<< HEAD
=======
├── pretrainig.py       # Pretraining functions for DeepIFSAC.
>>>>>>> a13f22d... DeepIFSAC pipeline [OpenML]
├── my_train.py                     # Main training script.
├── environment.yaml                # Environment configuration file.
└── README.md                       # This file.
```

---
<<<<<<< HEAD
=======
>>>>>>> Initial commit: Add readme.md
>>>>>>> a13f22d... DeepIFSAC pipeline [OpenML]
