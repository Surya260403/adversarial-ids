# Adversarial Machine Learning Against Network Intrusion Detection Systems (IDS)

<p align="center">
  A Cybersecurity Class Project exploring machine learning-based Network Intrusion Detection Systems and their robustness against controlled adversarial feature perturbations.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-Cybersecurity-blue" alt="Cybersecurity Project">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Completed">
</p>

## Overview

This project investigates the robustness of Machine Learning-based Network Intrusion Detection Systems (NIDS) against controlled adversarial feature perturbations.

Using the CICIDS2017 dataset, the project evaluates five machine learning algorithms under clean and adversarial conditions. It also explores adversarial training as a strategy for improving detection robustness.

The project includes a Streamlit dashboard for exploring model performance, algorithm comparisons, and adversarial stress-test results.

## Objectives

- Develop machine learning-based network intrusion detection models.
- Compare the performance of five classification algorithms.
- Evaluate model performance on clean test data.
- Investigate the impact of controlled adversarial feature perturbations.
- Explore adversarial training to improve robustness.
- Present experimental results through an interactive dashboard.

## Machine Learning Algorithms

The project evaluates five algorithms:

1. Baseline Random Forest
2. Robust Random Forest (adversarial augmentation)
3. XGBoost
4. Linear Support Vector Machine (SVM)
5. Multi-Layer Perceptron (MLP)

## Dataset

**Dataset:** CICIDS2017

The experiments use a balanced dataset containing:

- 500,000 rows
- 78 features
- Grouped train/test splitting

The original dataset is not included in this repository. Users must obtain the CICIDS2017 dataset separately and follow its applicable usage and licensing conditions.

## Experimental Methodology

### 1. Data Processing

The dataset is processed and prepared for binary intrusion detection classification.

### 2. Clean Model Evaluation

The five machine learning algorithms are evaluated using clean test data.

### 3. Adversarial Perturbation

Controlled feature-space perturbations are applied to continuous features to investigate their effect on model performance.

The robustness experiment uses:

- 42 continuous features
- IQR-scaled perturbations
- Four perturbation levels: 1%, 3%, 5%, and 10%
- Five random seeds

**Important:** These experiments use synthetic feature-space perturbations. They do not represent packet-level attacks or direct attacks against real network infrastructure.

### 4. Adversarial Training

A Robust Random Forest is trained using adversarial augmentation and evaluated to investigate improvements in robustness.

## Experimental Results

### Clean Test Accuracy

| Algorithm | Accuracy |
|---|---:|
| Baseline Random Forest | 99.871% |
| Robust Random Forest | 99.884% |
| XGBoost | 99.928% |
| Linear SVM | 95.134% |
| MLP | 98.739% |

**Key finding:** XGBoost achieved the highest clean-test accuracy among the evaluated models.

### Adversarial Robustness

The Robust Random Forest achieved approximately 99.99% attack detection across the tested synthetic perturbation strengths.

These findings apply to the experimental setup and perturbation conditions used in this project. They should not be interpreted as universal immunity against adversarial attacks.

## Interactive Dashboard

The project includes a Streamlit dashboard with the following sections:

- Executive Overview
- Algorithm Comparison
- Baseline IDS
- Robustness Comparison
- Adversarial Stress Test
- Methodology

Dashboard screenshots will be added in the `docs/screenshots/` directory.

## Project Structure

```text
adversarial-ids/
│
├── dashboard.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/                  # Dataset (not included in GitHub)
├── models/                # Trained models (not included)
├── docs/
│   └── screenshots/
│
├── reports/
│   └── Final_Project_Report.md
│
├── results/
│   ├── model_comparison.csv
│   ├── robustness_comparison.csv
│   ├── robustness_validation.csv
│   └── ...
│
├── src/
│   ├── data_pipeline.py
│   ├── compare_models.py
│   ├── robustness_experiment.py
│   ├── adversarial_evaluation.py
│   └── ...
│
└── tests/
```

## Installation and Setup

### Prerequisites

- Python 3.x
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Surya260403/adversarial-ids.git
cd adversarial-ids
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The dashboard and model comparison may also require Streamlit, Plotly, and XGBoost. Install them if they are not already included in your environment:

```bash
pip install streamlit plotly xgboost
```

### 4. Run the dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser.

**Note:** The original CICIDS2017 dataset and trained model files are not included in this public repository. Some project functionality may require the corresponding local data and model artifacts.

## Safety and Ethical Considerations

All experiments are dataset-based and controlled.

This project does not perform attacks against real systems, networks, or third-party infrastructure.

The adversarial experiments are conducted in a controlled feature space for academic research and robustness evaluation.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- Joblib

## Project Report

The detailed project report is available here:

[`reports/Final_Project_Report.md`](reports/Final_Project_Report.md)

## Contributors

**Group G — Cybersecurity Class Project**

- Aryan Chand
- Pratyush Ranjan Behera
- Rajeswari Sahoo
- Surya Narayan Mohanta
- Muskan Devi
- Subasish Saha

## License

See the [`LICENSE`](LICENSE) file for licensing information.

---

<p align="center">
  Developed as an academic cybersecurity project.
</p>