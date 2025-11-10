# 1. DeepIFSAC
This is the official implementation of paper titled, "DeepIFSAC: Deep Imputation of Missing Values Using Feature and Sample Attention within Contrastive Framework".

---

## 📖 Paper  
If you use this repository, please cite our paper:  

> Kowsar, I., Rabbani, S. B., Hou, Y., & Samad, M. D. (2025).  
> **DeepIFSAC: Deep imputation of missing values using feature and sample attention within contrastive framework.**  
> *Knowledge-Based Systems, 318,* 113506.  
> [https://doi.org/10.1016/j.knosys.2025.113506](https://doi.org/10.1016/j.knosys.2025.113506)

---

## 📌 Citation (BibTeX)  

```bibtex
@article{kowsar2025deepifsac,
  title={DeepIFSAC: Deep imputation of missing values using feature and sample attention within contrastive framework},
  author={Kowsar, Ibna and Rabbani, Shourav B and Hou, Yina and Samad, Manar D},
  journal={Knowledge-Based Systems},
  volume={318},
  pages={113506},
  year={2025},
  publisher={Elsevier}
}
```
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

DeepIFSAC is designed to work with tabular datasets (e.g., from OpenML). The model implements both a **pretraining phase** (with options for contrastive and denoising modules along with advanced data augmentations like CutMix and MixUp) and a **finetuning phase** that trains downstream classifiers.

---

## 4. Installation

### 4.1 Clone the Repository

```bash
git clone https://github.com/mdsamad001/DeepIFSAC.git
cd DeepIFSAC
```

### 4.2 Create a Virtual Environment (optional but recommended)

If you use Conda, create an environment using the provided YAML file:

```bash
conda env create -f difsac_env.yml
conda activate DeepIFSAC
```

Ensure that you have PyTorch installed with the proper CUDA version if you intend to use a GPU. All dependencies and their versions are specified in the `environment.yml` file.

---

## 5. Dataset

The code leverages a dataset loading function (`my_data_prep_openml`) located in the `data_openml` module. By providing a dataset ID (`--dset_id`), the code automatically downloads and processes the dataset from OpenML.

---

## 6. Training the Model

The training process is divided into two main phases: **pretraining (Imputation)** and **downstream finetuning (Classification)**.

### 6.1 Pretraining (Imputation)

DeepIFSAC can be pretrained using various objectives (e.g., denoising, contrastive loss). To run pretraining, set the `--pretrain` flag and specify additional parameters (like number of pretrain epochs, augmentation type, missing rate, etc.). The pretraining function (`DeepIFSAC_pretrain`) takes care of data augmentation, computes losses over epochs, and saves training metrics.

### 6.2 Finetuning/Downstream Evaluation (Classification)

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
For DeepIFSAC with contrastive --attentiontype = colorow which takes default pt_tasks = ['denoising', 'contrastive'] and for DeepIFSAC without contrastive, --attentiontype = colorowatt with pt_tasks = ['denoising']
```bash
python my_train.py \
  --dset_id 11 \
  --task multiclass \
  --attentiontype colrow \
  --pretrain \
  --pretrain_epochs 1000 \
  --epochs 200 \
  --batchsize 128 \
  --dset_seed 0 \
  --cuda_device 0 \
  --use_default_model \
  --missing_rate 0.5 \
  --missing_type mcar \
  --pt_aug cutmix
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
│   ├── pretrainmodel.py                # Implementation of the DeepIFSAC model.
│   ├── model.py                    # Additional models (e.g., simple_MLP).
│   └── ...
├── utils
│   └── ...                         # Helper functions for training, evaluation, etc.
├── augmentations
│   └── ...                         # Data augmentation routines.
├── pretraining.py       # Pretraining functions for DeepIFSAC.
├── my_train.py                     # Main training script.
├── environment.yaml                # Environment configuration file.
└── README.md                       # This file.
```
