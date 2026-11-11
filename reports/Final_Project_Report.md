# ADVERSARIAL MACHINE LEARNING AGAINST NETWORK INTRUSION DETECTION SYSTEMS

## Final Year Project Report

---

**Submitted by:** Surya

**Degree:** [BCA / MCA — Please specify]

**College:** [College Name]

**Department:** [Department of Computer Science / Information Technology]

**Academic Year:** [Academic Year]

---

---

## CERTIFICATE

*This is to certify that the project entitled*

**"ADVERSARIAL MACHINE LEARNING AGAINST NETWORK INTRUSION DETECTION SYSTEMS"**

*is a bonafide record of work carried out by* **Surya** *in partial fulfilment of the requirements for the award of the degree of* **[BCA/MCA]** *from* **[University Name]**, *during the academic year* **[Academic Year]**.

| | |
|---|---|
| **Project Guide:** | ________________________ |
| | [Name], [Designation] |
| | [Department] |
| | [College Name] |

| | |
|---|---|
| **Head of Department:** | ________________________ |
| | [Name], [Designation] |
| | Department of [CS/IT] |
| | [College Name] |

**Date:** _______________

**Place:** _______________

---

---

## DECLARATION

I, **Surya**, hereby declare that the project entitled **"Adversarial Machine Learning Against Network Intrusion Detection Systems"** submitted in partial fulfilment of the requirements for the award of the degree of **[BCA/MCA]** at **[College Name]**, affiliated to **[University Name]**, is an original and independent piece of work carried out by me under the guidance of **[Supervisor Name]**.

I further declare that this work has not been previously submitted, in whole or in part, to any other university or institution for the award of any degree or examination.

All sources of information and data used in this report have been duly acknowledged. The experimental results presented in this report are actual outputs of the implemented system and have not been fabricated or altered.

**Signature:** _______________________

**Name:** Surya

**Date:** _______________

**Place:** _______________

---

---

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude to my project guide, **[Supervisor Name]**, for their invaluable guidance, constant encouragement, and constructive suggestions throughout the course of this project. Their insights into machine learning security and adversarial robustness were instrumental in shaping the direction and depth of this research.

I extend my heartfelt thanks to the **Head of the Department**, **[HOD Name]**, for providing the necessary facilities and an encouraging environment for undertaking this project.

I am grateful to the faculty members of the Department of **[CS/IT]** at **[College Name]** for their academic support and encouragement throughout my programme.

I would also like to acknowledge the **Canadian Institute for Cybersecurity** for making the CICIDS2017 dataset publicly available, which served as the primary experimental dataset for this project.

Finally, I thank my family and friends for their moral support and patience during the course of this work.

**Surya**
[Academic Year]

---

---

## ABSTRACT

Network Intrusion Detection Systems (IDS) are a cornerstone of modern cybersecurity infrastructure, responsible for identifying malicious network traffic in real time. The increasing deployment of machine learning models within IDS introduces a new class of vulnerability: adversarial perturbations — small, crafted modifications to input data that can mislead a classifier into misidentifying attack traffic as benign.

This project investigates the **adversarial robustness of five machine learning classifiers** applied to network intrusion detection, using the publicly available **CICIDS2017 dataset**. The dataset was preprocessed and balanced to yield 500,000 records (399,996 training; 100,004 testing), with binary labels distinguishing benign and attack traffic.

Five classification algorithms were implemented and evaluated: **Baseline Random Forest**, **Robust Random Forest** (trained with adversarial data augmentation), **XGBoost**, **Linear SVM**, and **Neural Network (MLP)**. On the clean test set, XGBoost achieved the highest accuracy of **99.928%**, while the Robust Random Forest achieved **99.884%**.

An adversarial perturbation framework was designed and applied uniformly across all five models. Perturbations are generated as **IQR-scaled uniform random noise** applied exclusively to 42 continuous network flow features, respecting training-derived feature boundary constraints. Four perturbation strengths — 1%, 3%, 5%, and 10% of IQR — were evaluated across five independent random seeds (42, 101, 2024, 777, 9999).

The experimental results demonstrate that the **Robust Random Forest is the most adversarially resilient model**, maintaining an attack detection rate above **99.99%** at all perturbation strengths. The Baseline Random Forest is the most vulnerable, with detection rates falling to **62.04%** at 10% perturbation. XGBoost, Linear SVM, and MLP exhibit intermediate robustness, with MLP showing notably stable detection rates (97.32% at 10% perturbation).

The project concludes that adversarial data augmentation during training is an effective defence strategy for tree-based IDS models, and that perturbation robustness must be evaluated alongside clean accuracy when selecting an IDS classifier for real-world deployment.

**Keywords:** Intrusion Detection System, Adversarial Machine Learning, Random Forest, XGBoost, Support Vector Machine, Neural Network, CICIDS2017, Adversarial Robustness, Feature Perturbation

---

---

## TABLE OF CONTENTS

1. Introduction
2. Problem Statement
3. Research Objectives
4. Scope of the Project
5. Literature Review
6. Dataset Description
7. System Architecture
8. Methodology and Implementation
9. Data Preprocessing
10. Adversarial Perturbation Methodology
11. Robust Random Forest Training
12. Experimental Setup
13. Performance Metrics
14. Experimental Results and Analysis
15. Discussion
16. Limitations
17. Future Work
18. Conclusion
19. References
20. Appendix

---

---

## 1. INTRODUCTION

### 1.1 Network Security and the Importance of Intrusion Detection

The rapid proliferation of internet-connected devices and the increasing digitisation of critical infrastructure have made computer network security one of the most important challenges of the modern era. Organisations across sectors — including finance, healthcare, government, and education — routinely face threats including unauthorised access, data exfiltration, denial of service attacks, and malware propagation. The consequences of a successful network intrusion can be severe, ranging from financial loss and reputational damage to compromise of national security systems.

A **Network Intrusion Detection System (IDS)** is a monitoring tool that analyses network traffic to identify patterns associated with known attack types or anomalous behaviour. IDS can be deployed in one of two modes: **signature-based detection**, which compares traffic against a database of known attack signatures, and **anomaly-based detection**, which learns a model of normal network behaviour and flags deviations from that model. While signature-based systems are reliable for known threats, they are inherently unable to detect novel attacks. Anomaly-based IDS, particularly those powered by machine learning, offer the promise of detecting previously unseen attack patterns.

### 1.2 Machine Learning in IDS

The integration of machine learning into intrusion detection has been an active research area for over two decades. Machine learning classifiers can be trained on labelled network traffic data to distinguish benign connections from various attack categories. Algorithms such as Decision Trees, Random Forests, Support Vector Machines, Gradient Boosting methods, and Neural Networks have all been applied to this domain, often achieving very high accuracy on benchmark datasets.

However, high accuracy on a held-out test set does not guarantee reliable performance in a real-world adversarial environment. Attackers may deliberately craft network traffic to evade detection, exploiting weaknesses in the classifier's learned decision boundary.

### 1.3 Adversarial Machine Learning

**Adversarial Machine Learning** is the study of attacks on and defences for machine learning systems. An adversarial example is an input that has been subtly modified — often in a way that is imperceptible or operationally insignificant — to cause a machine learning model to produce an incorrect output. This concept was formalised in the image classification domain (Szegedy et al., 2013; Goodfellow et al., 2015), but the fundamental vulnerability applies equally to tabular data classifiers such as those used in IDS.

In the network security domain, adversarial perturbations correspond to modifications of network flow statistics — such as packet lengths, inter-arrival times, and byte counts — that may cause an attack flow to be misclassified as benign, thereby evading detection.

### 1.4 Motivation for this Research

Despite the extensive literature on machine learning IDS, comparatively few studies evaluate multiple classifiers under identical, controlled adversarial conditions using the same dataset, perturbation framework, and evaluation protocol. This project addresses that gap by:

1. Implementing five diverse classifiers on a large, balanced version of the CICIDS2017 dataset.
2. Designing a principled, feature-aware adversarial perturbation framework that respects the statistical properties of real network flows.
3. Evaluating all five classifiers under identical perturbation conditions.
4. Training a Robust Random Forest using adversarial data augmentation as a defensive baseline.
5. Comparing clean and adversarial performance to identify the most suitable classifier for robust network IDS deployment.

---

---

## 2. PROBLEM STATEMENT

Machine learning classifiers achieve high accuracy on clean, held-out network traffic datasets. However, their behaviour under adversarial conditions — where an attacker deliberately modifies the statistical properties of malicious traffic flows to evade detection — remains poorly characterised in multi-model comparative studies.

Specifically, this project addresses the following problem:

> *Given five diverse machine learning classifiers trained on the CICIDS2017 network traffic dataset, how does each classifier's attack detection capability degrade when network flow features are subjected to controlled adversarial perturbations of varying strengths? Can adversarial data augmentation during training produce a classifier that maintains high detection rates under such perturbations?*

The perturbations studied are **IQR-scaled uniform random noise** applied to continuous network flow statistics, constrained to remain within the realistic operating range of the training data. This models a scenario in which an attacker has the ability to modify continuous flow-level statistics (for example, by adjusting timing or payload size) while preserving discrete, structural properties of the protocol.

---

---

## 3. RESEARCH OBJECTIVES

The specific objectives of this project are:

1. **To implement and evaluate** five machine learning classifiers — Baseline Random Forest, Robust Random Forest, XGBoost, Linear SVM, and Neural Network (MLP) — on a large balanced version of the CICIDS2017 dataset.

2. **To design a principled adversarial perturbation framework** that modifies only continuous network flow features, respects realistic feature boundaries derived from training data, and applies identical perturbations across all five models.

3. **To evaluate the adversarial robustness** of each classifier at perturbation strengths of 1%, 3%, 5%, and 10% of the IQR, across five independent random seeds.

4. **To train a Robust Random Forest** using adversarial data augmentation and assess whether this training strategy improves resilience to test-time perturbations.

5. **To quantify attack detection rate and evasion rate** as primary adversarial robustness metrics, in addition to standard classification metrics (accuracy, precision, recall, F1-score, false positive rate).

6. **To develop an interactive research dashboard** that presents the clean and adversarial performance of all five models in a visually accessible and reproducible manner.

7. **To identify the most suitable classifier** for robust network IDS deployment, balancing clean accuracy and adversarial resilience.

---

---

## 4. SCOPE OF THE PROJECT

This project is scoped as follows:

**In Scope:**
- Binary classification of network traffic as benign or attack using the CICIDS2017 dataset.
- Training and evaluation of five machine learning classifiers.
- Design, implementation, and evaluation of a feature-space adversarial perturbation framework.
- Adversarial data augmentation for the Robust Random Forest.
- Comparative evaluation of all five models under identical adversarial conditions.
- Interactive Streamlit dashboard for result visualisation.
- Academic reporting of methodology and experimental findings.

**Out of Scope:**
- Packet-level adversarial attacks (e.g., modification of raw byte sequences in PCAP files).
- Gradient-based adversarial attack algorithms (e.g., FGSM, PGD, C&W) — the implemented perturbations are IQR-scaled uniform random noise, not gradient-guided.
- Multi-class classification of specific attack types (the project uses binary benign/attack classification).
- Real-time deployment or integration with live network monitoring infrastructure.
- Attacks targeting the training process (data poisoning, backdoor attacks).
- Defence mechanisms other than adversarial data augmentation (e.g., certified defences, input pre-processing, ensemble defences).

**Limitation of perturbation realism:** The adversarial perturbations implemented in this project modify continuous network flow statistics in feature space. While this simulates the effect of an adversary manipulating flow-level statistics, it does not model the physical constraints of generating such traffic at the packet level. The feasibility of translating specific feature-space perturbations into actual network packets was not evaluated.

---

---

## 5. LITERATURE REVIEW

### 5.1 Network Intrusion Detection Systems

Intrusion Detection Systems have been studied extensively since the late 1980s. Denning (1987) proposed one of the earliest anomaly-based IDS frameworks, modelling normal user and system behaviour to detect deviations. Subsequent work established the two principal paradigms — signature-based and anomaly-based detection — that continue to define the field.

The emergence of machine learning offered a new approach to anomaly detection: rather than manually specifying rules, a classifier is trained on labelled examples of benign and malicious traffic. The KDD Cup 1999 dataset (Stolfo et al., 1999) became the first widely used benchmark, enabling reproducible comparisons across algorithms. However, its known limitations (class imbalance, feature redundancy, and outdated attack scenarios) motivated the development of more realistic benchmarks.

### 5.2 The CICIDS2017 Dataset

Sharafaldin, Lashkari, and Ghorbani (2018) introduced the **CICIDS2017 dataset**, generated by the Canadian Institute for Cybersecurity. CICIDS2017 captures five days of network traffic, including benign activity alongside seven modern attack categories: DoS, DDoS, Web Attacks, Infiltration, Botnet, Brute Force, and PortScan. The dataset provides 78 network flow features extracted by the CICFlowMeter tool. Its realistic attack scenarios, labelled ground truth, and feature richness have made it one of the most widely used IDS benchmarks in current research.

### 5.3 Machine Learning Classifiers in IDS

**Random Forests** (Breiman, 2001) are ensemble classifiers that aggregate predictions from multiple independently trained decision trees. Their robustness to overfitting, ability to handle high-dimensional feature spaces, and interpretability through feature importance scores have made them a popular baseline in IDS research.

**XGBoost** (Chen and Guestrin, 2016) is a gradient boosting framework that constructs an ensemble of decision trees sequentially, with each tree correcting the residual errors of its predecessors. Its regularisation mechanisms, support for sparse data, and computational efficiency have led to consistently strong performance on tabular classification tasks.

**Support Vector Machines** (Cortes and Vapnik, 1995) seek a maximum-margin hyperplane that separates classes in a high-dimensional feature space (or its kernel-induced transformation). Linear SVMs are particularly well suited to linearly separable or approximately separable problems and are computationally tractable on large datasets.

**Neural Networks (Multi-Layer Perceptrons)** are universal function approximators that learn hierarchical representations of input data through layers of non-linear transformations. While deeper architectures (CNNs, RNNs) are common in image and sequence tasks, MLP architectures remain effective for tabular network traffic data.

### 5.4 Adversarial Machine Learning

Szegedy et al. (2013) first demonstrated that small, imperceptible perturbations to image inputs could cause deep neural networks to misclassify them with high confidence. Goodfellow et al. (2015) introduced the **Fast Gradient Sign Method (FGSM)**, a computationally efficient approach to generating adversarial examples by perturbing inputs in the direction of the loss gradient. Subsequent gradient-based attacks, including Projected Gradient Descent (PGD) (Madry et al., 2018), have become standard benchmarks for evaluating model robustness.

However, gradient-based attacks are most naturally defined for differentiable models (e.g., neural networks) and require access to model gradients. Their direct application to tree-based classifiers is non-trivial. Moreover, in the network IDS domain, the relationship between feature-space perturbations and physically realisable network packets introduces additional constraints.

Rubinstein et al. (2009) demonstrated that an attacker can manipulate network traffic statistics to evade anomaly-based IDS, establishing the relevance of adversarial considerations in this domain. Subsequent work (e.g., Corona et al., 2013; Apruzzese et al., 2019) has studied various evasion strategies against network IDS classifiers.

**Adversarial training** — augmenting the training set with perturbed examples — is one of the most widely studied defence strategies (Madry et al., 2018). The intuition is that if the model sees adversarial examples during training, it learns a more robust decision boundary. This principle underlies the Robust Random Forest implemented in this project.

### 5.5 Feature-Space Perturbations in Network IDS

A key challenge in adversarial IDS research is ensuring that perturbations are **semantically meaningful** and **operationally feasible**. Unlike images, network flow features have physical constraints: discrete features (flag counts, protocol identifiers) cannot be freely modified without changing protocol behaviour; continuous features (packet lengths, inter-arrival times) have natural bounds imposed by the underlying protocol and network conditions.

This project adopts the approach of restricting perturbations to continuous network flow features, with magnitudes bounded by the Interquartile Range (IQR) of each feature in the training data, and values clipped to remain within the 1st–99th percentile range of training observations. This approach is consistent with the feature-space perturbation methodology described in prior IDS adversarial robustness studies.

> **Note on References:** Standard academic references cited above (Breiman 2001, Chen & Guestrin 2016, Goodfellow et al. 2015, Sharafaldin et al. 2018) are verifiable in published literature. The student is advised to verify exact publication details before final submission and to supplement with any additional sources consulted during the project.

---

---

## 6. DATASET DESCRIPTION

### 6.1 Source: CICIDS2017

The **CICIDS2017 (Canadian Institute for Cybersecurity Intrusion Detection System 2017)** dataset was used as the primary experimental data source. It was generated by the Canadian Institute for Cybersecurity and captures five days of realistic network traffic, encompassing both normal (benign) connections and a range of modern attack scenarios.

**Attack categories present in CICIDS2017:**
- Benign (normal traffic)
- DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest
- DDoS
- PortScan
- Brute Force (FTP, SSH)
- Web Attacks (SQL Injection, XSS, Brute Force)
- Infiltration
- Botnet (ARES)

### 6.2 Binary Classification

For this project, the multi-class attack labels were reduced to a **binary classification problem**: traffic is labelled as either **Benign (0)** or **Attack (1)**. This simplification is appropriate for the research question, which focuses on whether an IDS can correctly distinguish attack traffic from benign traffic, and how this ability degrades under adversarial perturbation.

### 6.3 Feature Set

The dataset provides **78 features** extracted from network flows by the CICFlowMeter tool. These features characterise network connections at the flow level, including:

- **Temporal features:** Flow duration, inter-arrival times (IAT) in forward and backward directions.
- **Packet length features:** Maximum, minimum, mean, and standard deviation of packet lengths in both directions.
- **Traffic rate features:** Bytes per second, packets per second.
- **Flag-based features:** Counts of TCP flags (SYN, ACK, FIN, RST, PSH, URG, CWE, ECE).
- **Subflow and bulk features:** Subflow packet and byte counts, bulk transfer statistics.
- **Window features:** Initial TCP window sizes (forward and backward).
- **Activity features:** Active and idle duration statistics.
- **Miscellaneous:** Down/Up ratio, average packet size, segment sizes.

### 6.4 Dataset Preprocessing and Splitting

The raw CICIDS2017 data was preprocessed and a balanced subset of **500,000 records** was selected to avoid the severe class imbalance present in the original dataset. The dataset was split into training and testing subsets using a **grouped split strategy** that ensures samples from the same network session are not split across train and test sets, thereby preventing data leakage.

### 6.5 Dataset Statistics

| Split | Total Records | Benign | Attack |
|---|---|---|---|
| Training | 399,996 | 199,939 | 200,057 |
| Testing | 100,004 | 50,061 | 49,943 |
| **Total** | **500,000** | **249,000** (approx.) | **250,000** (approx.) |

The dataset is approximately balanced between benign and attack classes in both the training and testing splits.

### 6.6 Feature Columns

The 78 features used in model training span the full range of CICFlowMeter statistics. The `Label` column contains the original multi-class label (e.g., "BENIGN", "DoS Hulk", etc.) and the `BinaryLabel` column contains the binary target (0 = benign, 1 = attack). Of the 78 features, **42 are continuous** and subject to adversarial perturbation; the remaining **36 are discrete or structural** and are left unchanged.

---

---

## 7. SYSTEM ARCHITECTURE

### 7.1 Overview

The project follows a modular pipeline architecture, from raw dataset input through preprocessing, model training, adversarial evaluation, and dashboard visualisation. The diagram below illustrates the high-level system architecture.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                              │
│                                                                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │ CICIDS2017   │───▶│  PREPROCESSING   │───▶│  GROUPED TRAIN/TEST    │  │
│  │ Raw Dataset  │    │  - Imputation    │    │  SPLIT                 │  │
│  │ (78 features)│    │  - Scaling       │    │  Train: 399,996 rows   │  │
│  │              │    │  - Binary label  │    │  Test:  100,004 rows   │  │
│  └──────────────┘    └──────────────────┘    └────────────┬───────────┘  │
│                                                            │              │
│           ┌────────────────────────────────────────────────┘              │
│           ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                     MODEL TRAINING                                   │ │
│  │                                                                      │ │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌──────┐  ┌──────┐  │ │
│  │  │ Baseline   │  │ Robust RF  │  │ XGBoost  │  │ SVM  │  │ MLP  │  │ │
│  │  │ Random     │  │(+ 60,000   │  │          │  │      │  │      │  │ │
│  │  │ Forest     │  │augmented   │  │          │  │      │  │      │  │ │
│  │  │            │  │ samples)   │  │          │  │      │  │      │  │ │
│  │  └────────────┘  └────────────┘  └──────────┘  └──────┘  └──────┘  │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                         │
│           ┌─────────────────────┘                                         │
│           ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │               CLEAN EVALUATION (src/compare_models.py)              │ │
│  │  - Accuracy, Precision, Recall, F1, FPR, Training Time             │ │
│  │  - Output: results/model_comparison.csv                             │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                         │
│           ┌─────────────────────┘                                         │
│           ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │       ADVERSARIAL PERTURBATION FRAMEWORK                            │ │
│  │  (src/compare_adversarial_robustness.py)                            │ │
│  │                                                                      │ │
│  │  Feature Bounds  ──▶  IQR-scaled uniform noise  ──▶  5 seeds       │ │
│  │  (42 continuous features)   ε ∈ {1%, 3%, 5%, 10%}   × 4 strengths  │ │
│  │                                                                      │ │
│  │  Applied identically to all 5 models                                │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                         │
│           ┌─────────────────────┘                                         │
│           ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                 ADVERSARIAL EVALUATION RESULTS                       │ │
│  │  results/five_model_adversarial_comparison.csv (205 rows)           │ │
│  │  results/five_model_confusion_matrices.csv    (105 rows)            │ │
│  │  results/five_model_summary_metrics.csv       ( 20 rows)            │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                         │
│           ┌─────────────────────┘                                         │
│           ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │             STREAMLIT INTERACTIVE DASHBOARD (dashboard.py)          │ │
│  │  Pages: Executive Overview │ Dataset │ Algorithm Comparison         │ │
│  │         Adversarial Robustness │ Methodology │ About                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Key Components

| Component | File | Purpose |
|---|---|---|
| Baseline RF Training | `src/compare_models.py` | Train XGBoost, SVM, MLP against existing RF |
| Robust RF Training | `src/robustness_experiment.py` | Train Robust RF with augmented data |
| Adversarial Comparison | `src/compare_adversarial_robustness.py` | Unified 5-model adversarial evaluation |
| Dashboard | `dashboard.py` | Interactive result visualisation |
| Feature Bounds | `results/perturbed_feature_bounds.csv` | IQR and percentile bounds for 42 features |

---

---

## 8. METHODOLOGY AND IMPLEMENTATION

### 8.1 Baseline Random Forest

**Algorithm:** Random Forest (Breiman, 2001) is an ensemble learning method that constructs multiple decision trees during training and outputs the mode (for classification) of the individual trees.

**Hyperparameters used:**
| Parameter | Value |
|---|---|
| Number of estimators (trees) | 100 |
| Max features per split | `sqrt` (square root of total features) |
| Random state | 42 |
| Other parameters | scikit-learn defaults |

**Role in this project:** The Baseline Random Forest serves as the primary conventional classifier baseline. It was trained on the grouped training set without any adversarial augmentation. Its training artefacts (imputer, feature column list) were saved and reused by the Robust Random Forest training pipeline. Because the existing Baseline RF model was reused from a prior training run, its exact training time was not measured in the current experiment.

**Stored artefacts:** `models/grouped_baseline_random_forest.joblib`, `models/grouped_baseline_imputer.joblib`, `models/grouped_feature_columns.joblib`

---

### 8.2 Robust Random Forest

**Algorithm:** Identical to the Baseline Random Forest in architecture (100 trees, `max_features="sqrt"`, `random_state=42`), but trained on an augmented dataset that includes synthetically perturbed attack samples.

**Training procedure (`src/robustness_experiment.py`):**
1. Load the grouped training set.
2. Identify attack samples in the training set.
3. Randomly sample up to `MAX_AUGMENTED_ATTACKS = 60,000` attack samples.
4. Apply IQR-scaled perturbations to the 42 continuous features of these samples (using ε = 5% IQR and seed 42).
5. Concatenate the original training data with the 60,000 perturbed attack samples.
6. Train a new Random Forest on the augmented dataset.

**Role in this project:** The Robust Random Forest represents the **adversarially trained defensive baseline**. By exposing the model to perturbed attack samples during training, the goal is to produce a classifier with a more robust decision boundary that is less easily fooled by test-time perturbations.

**Stored artefacts:** `models/robust_random_forest.joblib`, `models/robust_imputer.joblib`, `models/robust_feature_columns.joblib`

---

### 8.3 XGBoost (Extreme Gradient Boosting)

**Algorithm:** XGBoost (Chen and Guestrin, 2016) is a scalable gradient boosting framework. Unlike Random Forests, which train trees independently and in parallel, gradient boosting trains trees sequentially: each tree is fitted to the residual errors of the ensemble up to that point, with regularisation terms controlling model complexity.

**Hyperparameters used:**
| Parameter | Value |
|---|---|
| Number of estimators | 200 |
| Max depth | 6 |
| Learning rate | 0.1 |
| Tree method | `hist` (histogram-based, memory efficient) |
| Random state | 42 |
| Other parameters | XGBoost defaults |

**Role in this project:** XGBoost achieved the highest clean test accuracy (**99.928%**) among all five models. It serves as the benchmark for clean-data classification performance against which robustness trade-offs are evaluated.

**Stored artefacts:** `models/xgboost_model.joblib`

---

### 8.4 Linear SVM (Support Vector Machine)

**Algorithm:** A Support Vector Machine (Cortes and Vapnik, 1995) seeks a hyperplane that maximises the margin between the two classes. The **Linear SVM** variant uses a linear kernel — the decision boundary is a linear function of the input features. This is implemented via `sklearn.svm.LinearSVC`, which uses a primal formulation optimised with coordinate descent and is computationally tractable on large datasets.

**Hyperparameters used:**
| Parameter | Value |
|---|---|
| Regularisation parameter C | 1.0 |
| Max iterations | 5000 |
| Dual | `False` (primal formulation for large n_samples) |
| Random state | 42 |

**Preprocessing:** Features are **standardised** (zero mean, unit variance) using `sklearn.preprocessing.StandardScaler` before training. The fitted scaler is saved and applied at inference time.

**Role in this project:** Linear SVM provides a classical, well-understood linear baseline. Its relatively lower accuracy (95.134%) compared to tree-based models reflects the non-linearity of the network traffic classification problem.

**Stored artefacts:** `models/svm_model.joblib`, `models/svm_scaler.joblib`, `models/svm_imputer.joblib`

---

### 8.5 Neural Network (MLP — Multi-Layer Perceptron)

**Algorithm:** A Multi-Layer Perceptron is a feedforward neural network with one or more hidden layers of fully connected neurons with non-linear activation functions. During training, weights are updated via backpropagation using the Adam optimiser.

**Hyperparameters used:**
| Parameter | Value |
|---|---|
| Hidden layer sizes | (64, 32) — two hidden layers |
| Activation | ReLU |
| Solver | Adam |
| Batch size | 1024 |
| Max iterations | 30 |
| Early stopping | Enabled |
| Random state | 42 |

**Preprocessing:** Features are **standardised** (zero mean, unit variance) using `sklearn.preprocessing.StandardScaler` before training. The fitted scaler is saved and applied at inference time.

**Role in this project:** The MLP represents a neural network baseline for comparison with tree-based methods. With 98.739% clean accuracy, it occupies a middle position between the strong tree-based models (RF, XGBoost) and the linear SVM.

**Stored artefacts:** `models/mlp_model.joblib`, `models/mlp_scaler.joblib`, `models/mlp_imputer.joblib`

---

---

## 9. DATA PREPROCESSING

### 9.1 Missing Value Handling

The CICIDS2017 feature extraction process (CICFlowMeter) can produce undefined values in certain flow statistics (for example, division-by-zero in rate calculations). These values appear as `NaN` or `Inf` in the dataset. The preprocessing pipeline handles them as follows:

- **Infinite values** (`+Inf`, `-Inf`) are replaced with `NaN`.
- **NaN values** are imputed using the **median** of the respective feature column, computed from the training data only.
- The fitted imputer is serialised to disk so that the identical imputation is applied to all subsequent data, including test data and adversarial perturbation experiments.

### 9.2 Feature Selection and Preparation

The `Label` and `BinaryLabel` columns are separated from the feature matrix before model training. The feature matrix consists of **78 columns** (all CICFlowMeter statistics). The feature column list is saved as `models/grouped_feature_columns.joblib` to ensure consistent column ordering across training and inference.

### 9.3 Scaling for SVM and MLP

Decision tree-based models (Random Forest, XGBoost) are invariant to monotonic feature transformations and do not require feature scaling. However, the **Linear SVM** and **MLP** are sensitive to feature magnitude differences and require standardisation.

A `StandardScaler` is fitted on the training data for each of these models and saved to disk. During inference and during the adversarial evaluation, the scaler is applied **after** perturbation is applied in the original feature space, to replicate the conditions under which an attacker would perturb raw traffic statistics before they are processed by the model's pipeline.

### 9.4 Grouped Train-Test Split

A **grouped split** strategy was used to prevent data leakage between the training and testing sets. Network flows belonging to the same session or time window are assigned entirely to either the training or the testing set, not split across both. This prevents the classifier from exploiting temporal correlations between training and test samples, producing a more realistic estimate of generalisation performance.

The final split yielded:
- **Training set:** 399,996 records
- **Testing set:** 100,004 records

### 9.5 Leakage Prevention

All preprocessing statistics (imputer medians, scaler means and variances, feature boundary percentiles and IQR values) are computed **exclusively from the training set** and applied to the test set without re-fitting. This ensures that test-set information does not leak into the preprocessing pipeline, preserving the integrity of the evaluation.

---

---

## 10. ADVERSARIAL PERTURBATION METHODOLOGY

### 10.1 What is Adversarial Perturbation?

Adversarial perturbation refers to the deliberate modification of input features to cause a machine learning model to produce an incorrect prediction. In the context of network IDS, an adversary may manipulate the statistical properties of their malicious network traffic — such as adjusting packet timing, sizes, or rates — to make an attack flow resemble benign traffic, thereby evading detection.

### 10.2 Why is Perturbation Testing Required?

Standard evaluation of IDS classifiers on clean, unmodified test data does not reveal how well the model performs when an intelligent adversary actively attempts to evade it. A model that achieves near-perfect accuracy on clean data may be trivially fooled by even small perturbations. Adversarial perturbation testing quantifies this vulnerability, providing a more realistic assessment of model suitability for real-world deployment.

### 10.3 Feature Categorisation

Of the 78 features in the CICIDS2017 dataset, features are divided into two categories for the purpose of perturbation:

**Perturbed (42 continuous features):** These are flow-level statistics that naturally take continuous numerical values and can plausibly be varied by an attacker through adjustments to packet timing, payload sizes, or traffic rates. Examples include flow duration, packet length statistics, inter-arrival times, and byte/packet rates.

**Unperturbed (36 discrete or structural features):** These include TCP flag counts (SYN, ACK, FIN, RST, PSH, URG, CWE, ECE), protocol identifiers, and other integer-valued or binary features whose modification would require structural changes to the protocol behaviour, making free perturbation operationally implausible. These features are left unchanged.

### 10.4 The 42 Perturbed Features

The following 42 continuous features are subject to adversarial perturbation, as defined in `results/perturbed_feature_bounds.csv`:

| # | Feature | # | Feature |
|---|---|---|---|
| 1 | Flow Duration | 22 | Bwd IAT Mean |
| 2 | Fwd Packet Length Max | 23 | Bwd IAT Std |
| 3 | Fwd Packet Length Min | 24 | Bwd IAT Max |
| 4 | Fwd Packet Length Mean | 25 | Bwd IAT Min |
| 5 | Fwd Packet Length Std | 26 | Fwd Packets/s |
| 6 | Bwd Packet Length Max | 27 | Bwd Packets/s |
| 7 | Bwd Packet Length Min | 28 | Min Packet Length |
| 8 | Bwd Packet Length Mean | 29 | Max Packet Length |
| 9 | Bwd Packet Length Std | 30 | Packet Length Mean |
| 10 | Flow Bytes/s | 31 | Packet Length Std |
| 11 | Flow Packets/s | 32 | Packet Length Variance |
| 12 | Flow IAT Mean | 33 | Down/Up Ratio |
| 13 | Flow IAT Std | 34 | Average Packet Size |
| 14 | Flow IAT Max | 35 | Avg Fwd Segment Size |
| 15 | Flow IAT Min | 36 | Avg Bwd Segment Size |
| 16 | Fwd IAT Total | 37 | Active Mean |
| 17 | Fwd IAT Mean | 38 | Active Max |
| 18 | Fwd IAT Std | 39 | Active Min |
| 19 | Fwd IAT Max | 40 | Idle Mean |
| 20 | Fwd IAT Min | 41 | Idle Max |
| 21 | Bwd IAT Total | 42 | Idle Min |

### 10.5 Perturbation Formula

For each perturbed feature, the adversarial value is computed as:

$$x' = x + \eta \cdot (\varepsilon \cdot \text{IQR})$$

where:
- $x$ is the original feature value of an attack sample
- $x'$ is the perturbed feature value
- $\varepsilon \in \{0.01, 0.03, 0.05, 0.10\}$ is the perturbation strength (1%, 3%, 5%, 10% of IQR)
- $\text{IQR}$ is the **Interquartile Range** (Q3 − Q1) of the feature, computed from the training data
- $\eta \sim \text{Uniform}(-1, 1)$ is an independent random noise scalar drawn for each feature of each sample

After applying the perturbation, each feature value is **clipped** to the range $[\text{P01}, \text{P99}]$, where P01 and P99 are the 1st and 99th percentile of the feature in the training data. This ensures that perturbed values remain within the realistic operating range observed during training.

$$x' = \text{clip}(x + \eta \cdot (\varepsilon \cdot \text{IQR}),\; \text{P01},\; \text{P99})$$

### 10.6 Perturbation Strengths

| Strength Label | ε value | Description |
|---|---|---|
| 1% | 0.01 | 1% of IQR added as noise |
| 3% | 0.03 | 3% of IQR added as noise |
| 5% | 0.05 | 5% of IQR added as noise |
| 10% | 0.10 | 10% of IQR added as noise |

### 10.7 Random Seeds

Five independent random seeds are used to assess the variability of perturbation effects:

**Seeds: 42, 101, 2024, 777, 9999**

For each combination of model, perturbation strength, and seed, a fresh set of random noise values $\eta$ is drawn. Mean and standard deviation of evaluation metrics are reported across the five seeds.

> **Note:** An additional seed value of 0 is used in the codebase solely for the clean baseline evaluation (measuring performance on unperturbed test data). This is distinct from the five perturbation seeds.

### 10.8 Attack-Only Perturbation

Perturbations are applied **exclusively to attack samples**. Benign samples are presented to the classifier unchanged. This models the threat scenario in which an attacker modifies only their own malicious traffic to evade detection, while benign users continue generating unmodified network activity.

The evaluation is performed on the **mixed test set** (containing both benign and perturbed-attack samples), which mirrors the realistic deployment condition of an IDS monitoring combined traffic.

### 10.9 Evaluation Metrics

For each (model, perturbation strength, seed) combination, the following metrics are computed:

- **Attack Detection Rate (ADR):** The fraction of attack samples correctly classified as attacks by the perturbed model. This is equivalent to the recall (sensitivity) on the attack class.

$$\text{ADR} = \frac{TP}{TP + FN}$$

- **Evasion Rate:** The fraction of attack samples that successfully evade detection (classified as benign after perturbation).

$$\text{Evasion Rate} = 1 - \text{ADR} = \frac{FN}{TP + FN}$$

- **Mixed Accuracy:** Overall classification accuracy on the mixed test set (benign + perturbed attacks).
- **Mixed F1-Score:** Weighted F1-score on the mixed test set.
- **False Positive Rate:** Rate at which benign samples are misclassified as attacks (unchanged by attack-only perturbation).

---

---

## 11. ROBUST RANDOM FOREST TRAINING

### 11.1 Motivation: Adversarial Data Augmentation

Adversarial data augmentation is a training-time defence strategy in which the model is exposed to adversarially perturbed examples during training. The rationale is straightforward: if the model encounters perturbed attack samples during training, it must learn a decision boundary that correctly classifies both clean and perturbed attacks as malicious. This encourages the model to learn more robust, generalised features of attack traffic rather than relying on precise feature values that an attacker could manipulate.

### 11.2 Augmentation Procedure

The Robust Random Forest training (`src/robustness_experiment.py`) proceeds as follows:

1. **Load** the grouped training dataset (399,996 samples).
2. **Identify** all attack samples in the training set (200,057 samples).
3. **Randomly sample** up to `MAX_AUGMENTED_ATTACKS = 60,000` attack samples from the training attack pool.
4. **Apply perturbations** to the 42 continuous features of these 60,000 samples using ε = 5% and seed 42.
5. **Concatenate** the original 399,996 training samples with the 60,000 perturbed attack samples to form an augmented training set of **459,996 samples**.
6. **Train** a new Random Forest (100 trees, `max_features="sqrt"`, `random_state=42`) on the augmented dataset.
7. **Save** the trained model, imputer, and feature column list to the `models/` directory.

### 11.3 Conventional vs Robust Random Forest

| Aspect | Baseline Random Forest | Robust Random Forest |
|---|---|---|
| Training data | 399,996 clean samples | 459,996 (clean + 60,000 perturbed attacks) |
| Training-time perturbations | None | ε = 5% IQR, seed 42 |
| Clean test accuracy | 99.871% | 99.884% |
| ADR at 1% perturbation | 72.88% | 99.99% |
| ADR at 10% perturbation | 62.04% | 99.99% |
| Model size (approx.) | 18.9 MB | 4.9 MB |

The remarkable improvement in adversarial robustness achieved by the Robust RF — from 62% to 99.99% detection rate at 10% perturbation — with only a marginal improvement in clean accuracy (99.871% → 99.884%) demonstrates the high effectiveness of adversarial training for tree-based classifiers in this setting.

The smaller model size of the Robust RF (4.9 MB vs 18.9 MB) is a consequence of the augmented training set creating a more generalised, less overfit tree structure with fewer nodes per tree.

---

---

## 12. EXPERIMENTAL SETUP

### 12.1 Software Environment

| Component | Details |
|---|---|
| Operating System | Windows 11 |
| Python Version | Python 3.11.9 (virtual environment) |
| Virtual environment | `e:\adversarial-ids\.venv\` |
| scikit-learn | For RF, SVM, MLP, preprocessing |
| XGBoost | For XGBoost classifier |
| NumPy | Numerical computations |
| Pandas | Data loading and manipulation |
| Joblib | Model serialisation |
| Streamlit | Interactive dashboard |
| Plotly | Dashboard visualisation |

### 12.2 Model Storage

All trained models and preprocessing artefacts are stored in the `models/` directory:

| Artefact | Purpose |
|---|---|
| `grouped_baseline_random_forest.joblib` | Baseline RF (trained on grouped split) |
| `grouped_baseline_imputer.joblib` | Imputer for Baseline RF |
| `grouped_feature_columns.joblib` | Feature column ordering |
| `robust_random_forest.joblib` | Adversarially trained Robust RF |
| `robust_imputer.joblib` | Imputer for Robust RF |
| `robust_feature_columns.joblib` | Feature columns for Robust RF |
| `xgboost_model.joblib` | Trained XGBoost classifier |
| `svm_model.joblib` | Trained Linear SVM |
| `svm_scaler.joblib` | StandardScaler for SVM |
| `svm_imputer.joblib` | Imputer for SVM |
| `mlp_model.joblib` | Trained MLP |
| `mlp_scaler.joblib` | StandardScaler for MLP |
| `mlp_imputer.joblib` | Imputer for MLP |

### 12.3 Evaluation Methodology

**Clean evaluation:** Each model is evaluated on the unperturbed 100,004-sample test set. Standard classification metrics are computed.

**Adversarial evaluation:** For each of the 20 (model × perturbation strength) combinations, and for each of the 5 random seeds, the attack samples in the test set are perturbed using the perturbation formula. The model is then evaluated on the mixed test set. This yields 5 × 4 × 5 = 100 evaluation instances per model (clean + 20 adversarial per seed per model), with mean and standard deviation reported across seeds.

### 12.4 Identical Test Conditions

A key design principle of the adversarial evaluation is that **all five models are evaluated on identical perturbed test samples**. For a given (perturbation strength, seed) pair, the same set of perturbed feature matrices is presented to all five models. This eliminates any confounding effect of different perturbation realisations and ensures that observed differences in detection rate are attributable solely to differences in model robustness.

---

---

## 13. PERFORMANCE METRICS

### 13.1 Confusion Matrix Terms

For binary classification (Attack = Positive, Benign = Negative):

| Term | Symbol | Definition |
|---|---|---|
| True Positive | TP | Attack correctly classified as Attack |
| True Negative | TN | Benign correctly classified as Benign |
| False Positive | FP | Benign incorrectly classified as Attack |
| False Negative | FN | Attack incorrectly classified as Benign (missed attack) |

### 13.2 Standard Classification Metrics

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$

### 13.3 Adversarial Robustness Metrics

$$\text{Attack Detection Rate (ADR)} = \frac{TP_{\text{perturbed}}}{TP_{\text{perturbed}} + FN_{\text{perturbed}}}$$

$$\text{Evasion Rate} = 1 - \text{ADR} = \frac{FN_{\text{perturbed}}}{TP_{\text{perturbed}} + FN_{\text{perturbed}}}$$

The ADR measures how many attack samples are still detected after perturbation. The Evasion Rate measures how many attack samples successfully evaded detection due to the perturbation.

**Statistical reporting:** Because five independent random seeds are used, the reported ADR and Evasion Rate are expressed as **mean ± standard deviation** across seeds, providing an indication of result stability and variance due to random noise realisations.

---

---

## 14. EXPERIMENTAL RESULTS AND ANALYSIS

### 14.1 Clean Test Performance

The following table shows the performance of all five models on the unperturbed test set.

| Model | Accuracy | Precision | Recall | F1-Score | FPR | FN (Missed Attacks) | FP (False Alarms) |
|---|---|---|---|---|---|---|---|
| Baseline RF | 99.871% | 99.886% | 99.856% | 99.871% | 0.114% | 72 | 57 |
| Robust RF | 99.884% | 99.892% | 99.876% | 99.884% | 0.108% | 62 | 54 |
| **XGBoost** | **99.928%** | 99.886% | **99.970%** | **99.928%** | 0.114% | **15** | 57 |
| Linear SVM | 95.134% | 92.904% | 97.721% | 95.252% | 7.447% | 1,138 | 3,728 |
| Neural Net (MLP) | 98.739% | 98.025% | 99.479% | 98.747% | 2.000% | 260 | 1,001 |

**Analysis:**
- **XGBoost** achieves the highest clean test accuracy (99.928%), missing only 15 attack samples and generating 57 false alarms on the 100,004-sample test set.
- Both **Random Forest variants** achieve similar high accuracy (~99.87–99.88%), demonstrating that adversarial augmentation does not degrade clean performance.
- **Linear SVM** produces substantially more false alarms (3,728 FP) and misses significantly more attacks (1,138 FN), reflecting the non-linear nature of the classification problem.
- **MLP** occupies an intermediate position (98.739% accuracy), generating 1,001 false alarms and missing 260 attacks.

**Training time:**
| Model | Training Time |
|---|---|
| Baseline RF | N/A (model reused from prior run) |
| Robust RF | N/A (model reused from prior run) |
| XGBoost | 7.81 seconds |
| Linear SVM | 150.57 seconds |
| MLP | 34.48 seconds |

**Prediction time:**
| Model | Prediction Time (ms per 1,000 samples) |
|---|---|
| Baseline RF | 5.31 ms |
| XGBoost | 0.94 ms |
| Linear SVM | 1.15 ms |
| MLP | 1.10 ms |

XGBoost is both the most accurate and the fastest at inference time, making it highly attractive for real-time IDS deployment based on clean performance alone.

---

### 14.2 Adversarial Evaluation: Attack Detection Rate

The following table reports the **mean attack detection rate (ADR)** and **standard deviation** across five random seeds for each model and perturbation strength.

| Perturbation | Baseline RF | Robust RF | XGBoost | Linear SVM | MLP |
|---|---|---|---|---|---|
| **1% IQR** | 72.88 ± 0.12% | **99.99 ± 0.00%** | 92.62 ± 0.05% | 95.12 ± 0.05% | 98.21 ± 0.01% |
| **3% IQR** | 65.17 ± 0.07% | **99.99 ± 0.00%** | 90.00 ± 0.03% | 92.76 ± 0.13% | 98.07 ± 0.02% |
| **5% IQR** | 63.69 ± 0.05% | **99.99 ± 0.00%** | 87.81 ± 0.03% | 89.77 ± 0.12% | 97.89 ± 0.02% |
| **10% IQR** | 62.04 ± 0.07% | **99.99 ± 0.00%** | 85.48 ± 0.06% | 84.00 ± 0.13% | 97.32 ± 0.04% |

**Key observations:**
1. The **Robust Random Forest** is essentially unaffected by perturbations at all tested strengths, maintaining > 99.99% ADR. Its adversarial training has produced near-perfect immunity to IQR-scaled uniform noise perturbations.
2. The **Baseline Random Forest** is the most vulnerable model, with ADR dropping to 62.04% at 10% perturbation — meaning over one-third of attack samples successfully evade detection after perturbation.
3. **MLP** shows notable adversarial stability, with ADR degrading only from 98.21% to 97.32% across all perturbation strengths. This suggests that the MLP's learned representations generalise reasonably well to perturbed inputs.
4. **XGBoost** shows moderate vulnerability, with ADR falling from 92.62% to 85.48% as perturbation increases.
5. **Linear SVM** initially outperforms XGBoost at low perturbation (95.12% vs 92.62% at 1%), but degrades more rapidly with increasing perturbation strength (84.00% vs 85.48% at 10%).

---

### 14.3 Adversarial Evaluation: Evasion Rate

| Perturbation | Baseline RF | Robust RF | XGBoost | Linear SVM | MLP |
|---|---|---|---|---|---|
| **1% IQR** | 27.12 ± 0.12% | **0.01 ± 0.00%** | 7.38 ± 0.05% | 4.88 ± 0.05% | 1.79 ± 0.01% |
| **3% IQR** | 34.83 ± 0.07% | **0.01 ± 0.00%** | 10.01 ± 0.03% | 7.24 ± 0.13% | 1.93 ± 0.02% |
| **5% IQR** | 36.31 ± 0.05% | **0.01 ± 0.00%** | 12.19 ± 0.03% | 10.23 ± 0.12% | 2.11 ± 0.02% |
| **10% IQR** | 37.96 ± 0.07% | **0.01 ± 0.00%** | 14.52 ± 0.06% | 16.00 ± 0.13% | 2.68 ± 0.04% |

**Key observations:**
1. The Baseline RF's evasion rate rises to nearly **38%** at 10% perturbation — a critically high value for a security-critical system.
2. The Robust RF's evasion rate remains effectively **0.01%** at all perturbation levels, confirming the efficacy of adversarial training.
3. MLP achieves the best evasion rate among non-adversarially-trained models, rising from only 1.79% at 1% to 2.68% at 10% — demonstrating the inherent robustness of neural network classifiers to this type of perturbation.
4. XGBoost and Linear SVM show comparable but higher evasion rates (7–16% range), with Linear SVM overtaking XGBoost's vulnerability at the 10% level.

---

### 14.4 Statistical Stability

The standard deviation of ADR and Evasion Rate across the five seeds is consistently **very small** (< 0.15% for all models at all perturbation levels), indicating that the results are highly stable and not sensitive to the specific random noise realisation. This validates the use of five seeds as sufficient for reliable estimation of adversarial performance.

---

### 14.5 Clean vs Adversarial Performance Trade-off

The following table compares clean accuracy against ADR at 10% perturbation, illustrating the robustness trade-off:

| Model | Clean Accuracy | ADR @ 10% | Recommendation |
|---|---|---|---|
| Baseline RF | 99.871% | 62.04% | Vulnerable — do not deploy without defence |
| **Robust RF** | **99.884%** | **99.99%** | **Best robust model — recommended for adversarial environments** |
| XGBoost | 99.928% | 85.48% | Best clean performance; moderate robustness |
| Linear SVM | 95.134% | 84.00% | Weakest clean performance; moderate robustness |
| MLP | 98.739% | 97.32% | Strong robustness without adversarial training |

---

---

## 15. DISCUSSION

### 15.1 Why is the Robust RF So Resilient?

The Robust Random Forest's near-perfect adversarial detection rate stems from its exposure to 60,000 perturbed attack samples during training. When the model learns from examples of attack traffic with modified feature values, it discovers more fundamental, robust patterns that distinguish attack traffic from benign — patterns that are invariant to the magnitude of continuous feature perturbation applied in this study.

However, it is important to note that the adversarial training used perturbations drawn from the same distribution as the test perturbations (IQR-scaled uniform noise). This means the Robust RF's immunity is specific to this perturbation type and magnitude range. It may not generalise to fundamentally different perturbation strategies, such as gradient-guided attacks or perturbations targeting discrete features.

### 15.2 Why is the Baseline RF Vulnerable?

Decision trees partition the feature space using threshold-based splits on individual features. When continuous features are perturbed, samples may cross these thresholds and be routed to different leaf nodes, changing the classification outcome. The Baseline RF, trained only on clean data, has learned thresholds that are closely calibrated to the clean feature distribution of attack traffic. Even small perturbations can push samples across decision boundaries in many trees simultaneously, particularly in regions of the feature space where benign and attack traffic are closely intermixed.

### 15.3 MLP's Unexpected Robustness

The MLP's strong adversarial performance (97.32% ADR at 10% perturbation) without any adversarial training is noteworthy. Neural networks with continuous activation functions and smooth decision boundaries may exhibit greater inherent robustness to small continuous perturbations compared to piece-wise constant tree-based classifiers. The MLP's representations, learned through gradient descent over many epochs, may capture more generalised, smooth features of attack traffic.

### 15.4 Implications for IDS Deployment

The experimental results have clear practical implications:

1. **Clean accuracy alone is an insufficient criterion for IDS classifier selection.** XGBoost's superior clean accuracy (99.928%) does not translate to superior adversarial robustness.
2. **Adversarial data augmentation is a highly effective defence** for tree-based classifiers, adding virtually no overhead to inference and negligibly affecting clean performance.
3. **MLP warrants further investigation** as a naturally robust classifier for adversarial IDS scenarios, potentially in combination with adversarial training.
4. **Linear SVM is not recommended** for adversarial IDS deployment due to its lower clean accuracy and moderate adversarial vulnerability.

---

---

## 16. LIMITATIONS

1. **Feature-space perturbations only:** The perturbations implemented in this project modify continuous feature values in the extracted flow-feature space. They do not correspond to actual packet-level modifications. The feasibility of translating specific feature-space perturbations into real network packets that satisfy all protocol constraints was not evaluated.

2. **IQR-scaled uniform noise — not gradient-based attacks:** The perturbation methodology uses IQR-scaled uniform random noise. This is not equivalent to gradient-based adversarial attacks (FGSM, PGD) which are specifically optimised to maximise classifier error. The evasion rates observed may be different under gradient-guided attacks.

3. **Robustness to perturbation type:** The Robust RF was trained and tested on the same perturbation distribution. Its robustness may not generalise to perturbations of a different type (e.g., targeted feature manipulation or structured evasion strategies).

4. **Binary classification only:** The project uses binary (benign/attack) labels. Multi-class performance (distinguishing attack types) was not evaluated.

5. **Single dataset:** Experiments are conducted only on the CICIDS2017 dataset. Generalisation to other network environments, traffic profiles, or attack scenarios is not established.

6. **No real-time deployment evaluation:** Model latency and throughput under real network traffic conditions were not measured; prediction times are based on batch evaluation on the test set.

7. **Discrete feature perturbations excluded:** Modifications to discrete features (flag counts, protocol identifiers) were not explored. Attackers who can manipulate these features may achieve higher evasion rates than observed here.

---

---

## 17. FUTURE WORK

1. **Gradient-based adversarial attacks:** Implement FGSM and PGD attacks against the differentiable models (MLP) to compare with the uniform noise perturbations used here, and explore surrogate-model-based attacks against the tree classifiers.

2. **Multi-class robustness evaluation:** Extend the analysis to multi-class classification, evaluating which attack types are most affected by adversarial perturbation and which are most robustly detected.

3. **Adversarial training for XGBoost and MLP:** Apply adversarial data augmentation to XGBoost and MLP training pipelines and evaluate the resulting improvement in adversarial robustness.

4. **Discrete feature perturbations:** Investigate the effect of modifying discrete, structural features (e.g., flag counts) and develop perturbation strategies that jointly modify continuous and discrete features while satisfying protocol constraints.

5. **Physical-level feasibility study:** Investigate the mapping from feature-space perturbations to packet-level modifications to establish the operational feasibility of the simulated adversarial scenarios.

6. **Alternative datasets:** Evaluate the same classifiers and perturbation framework on UNSW-NB15, CIC-IDS-2018, or other modern IDS benchmark datasets to assess generalisability.

7. **Certified defences:** Explore provably robust training approaches (randomised smoothing, interval bound propagation) for tree-based and neural network classifiers.

8. **Deployment integration:** Integrate the trained classifiers with a live network monitoring tool (e.g., Suricata, Zeek) to evaluate real-time performance.

---

---

## 18. CONCLUSION

This project has presented a comprehensive comparative study of five machine learning classifiers — Baseline Random Forest, Robust Random Forest, XGBoost, Linear SVM, and Neural Network (MLP) — for network intrusion detection, evaluated under both clean and adversarial conditions using the CICIDS2017 dataset.

On the clean test set, **XGBoost achieved the highest accuracy (99.928%)**, making it the preferred choice for clean-environment deployment. Both Random Forest variants achieved very high accuracy (~99.87–99.88%), while Linear SVM and MLP showed lower but still substantial accuracy (95.134% and 98.739%, respectively).

The adversarial evaluation revealed a stark divergence in model behaviour under IQR-scaled uniform perturbations. The **Baseline Random Forest, despite its high clean accuracy, is critically vulnerable** to adversarial perturbation, with attack detection rates falling to 62.04% at 10% perturbation strength. In contrast, the **Robust Random Forest — trained with adversarial data augmentation — maintains > 99.99% detection rate** at all perturbation strengths tested, demonstrating the effectiveness of adversarial training as a defence strategy.

The **MLP shows the best adversarial robustness among the non-adversarially-trained models** (97.32% ADR at 10% perturbation), suggesting that smooth neural network decision boundaries may confer natural resilience to continuous feature perturbations.

The key conclusion of this project is that **clean accuracy must not be the sole criterion for IDS classifier selection**. When an adversary actively manipulates network traffic statistics to evade detection, a classifier's adversarial robustness becomes the primary criterion. Adversarial data augmentation, implemented here via the Robust Random Forest, is a computationally inexpensive and highly effective technique to achieve near-perfect robustness without sacrificing clean performance.

Future work should investigate gradient-based attacks, multi-class robustness, adversarial training for gradient-based models, and the physical feasibility of feature-space perturbations to further strengthen the foundations of robust machine learning IDS.

---

---

## 19. REFERENCES

> **Note to student:** Verify the following references for completeness and accuracy before final submission. Add any additional sources consulted during the project.

1. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

2. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM.

3. **Cortes, C., & Vapnik, V.** (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273–297.

4. **Denning, D. E.** (1987). An Intrusion-Detection Model. *IEEE Transactions on Software Engineering*, SE-13(2), 222–232.

5. **Goodfellow, I. J., Shlens, J., & Szegedy, C.** (2015). Explaining and Harnessing Adversarial Examples. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*.

6. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A.** (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. In *Proceedings of the 6th International Conference on Learning Representations (ICLR 2018)*.

7. **Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A.** (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. In *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP 2018)* (pp. 108–116).

8. **Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R.** (2013). Intriguing Properties of Neural Networks. *arXiv preprint arXiv:1312.6199*.

9. **Rubinstein, B. I. P., Nelson, B., Huang, L., Joseph, A. D., Lau, S. H., Rao, S., Taft, N., & Tygar, J. D.** (2009). ANTIDOTE: Understanding and Defending Against Poisoning of Anomaly Detectors. In *Proceedings of the 9th ACM SIGCOMM Internet Measurement Conference* (pp. 1–14).

10. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

11. **Apruzzese, G., Colajanni, M., Ferretti, L., & Marchetti, M.** (2019). Addressing Adversarial Attacks Against Security Systems Based on Machine Learning. In *2019 11th International Conference on Cyber Conflict (CyCon)* (pp. 1–18). IEEE.

---

---

## 20. APPENDIX

### Appendix A: Project Directory Structure

```
e:\adversarial-ids\
├── data\
│   ├── train_grouped.csv          (399,996 rows — training set)
│   └── test_grouped.csv           (100,004 rows — test set)
├── models\
│   ├── grouped_baseline_random_forest.joblib
│   ├── grouped_baseline_imputer.joblib
│   ├── grouped_feature_columns.joblib
│   ├── robust_random_forest.joblib
│   ├── robust_imputer.joblib
│   ├── robust_feature_columns.joblib
│   ├── xgboost_model.joblib
│   ├── svm_model.joblib
│   ├── svm_scaler.joblib
│   ├── svm_imputer.joblib
│   ├── mlp_model.joblib
│   ├── mlp_scaler.joblib
│   └── mlp_imputer.joblib
├── results\
│   ├── model_comparison.csv
│   ├── model_confusion_matrices.csv
│   ├── perturbed_feature_bounds.csv       (42 features × IQR/P01/P99)
│   ├── robustness_validation.csv
│   ├── robustness_comparison.csv
│   ├── five_model_adversarial_comparison.csv   (205 rows)
│   ├── five_model_confusion_matrices.csv       (105 rows)
│   └── five_model_summary_metrics.csv          (20 rows)
├── src\
│   ├── compare_models.py
│   ├── robustness_experiment.py
│   ├── validate_robustness.py
│   ├── adversarial_evaluation_grouped.py
│   └── compare_adversarial_robustness.py
├── reports\
│   └── Final_Project_Report.md
├── dashboard.py
└── requirements.txt
```

### Appendix B: Perturbation Feature Bounds (Sample)

The following table shows the first 10 entries from `results/perturbed_feature_bounds.csv`, illustrating the IQR, P01, and P99 values used in the perturbation formula:

| Feature | IQR | P01 | P99 |
|---|---|---|---|
| Flow Duration | 60,731,812 | 3.0 | 117,349,040 |
| Fwd Packet Length Max | 315.0 | 0.0 | 2,065.0 |
| Fwd Packet Length Min | 6.0 | 0.0 | 68.0 |
| Fwd Packet Length Mean | 46.0 | 0.0 | 425.57 |
| Fwd Packet Length Std | 116.64 | 0.0 | 699.80 |
| Bwd Packet Length Max | 4,338.0 | 0.0 | 11,595.0 |
| Bwd Packet Length Min | 6.0 | 0.0 | 226.0 |
| Bwd Packet Length Mean | 1,650.43 | 0.0 | 2,901.75 |
| Bwd Packet Length Std | 1,757.58 | 0.0 | 5,795.50 |
| Flow Bytes/s | 93,629.77 | 0.0 | 10,300,000 |

### Appendix C: Key Source Code Files

| File | Purpose |
|---|---|
| [`src/compare_models.py`](file:///e:/adversarial-ids/src/compare_models.py) | Train and evaluate XGBoost, SVM, MLP |
| [`src/robustness_experiment.py`](file:///e:/adversarial-ids/src/robustness_experiment.py) | Train Robust RF with adversarial augmentation |
| [`src/validate_robustness.py`](file:///e:/adversarial-ids/src/validate_robustness.py) | Validate robustness of all 5 models |
| [`src/compare_adversarial_robustness.py`](file:///e:/adversarial-ids/src/compare_adversarial_robustness.py) | Full 5-model adversarial comparison experiment |
| [`dashboard.py`](file:///e:/adversarial-ids/dashboard.py) | Streamlit interactive dashboard |

---

*End of Report*

---

**Report prepared by:** Surya  
**Project:** Adversarial Machine Learning Against Network Intrusion Detection Systems  
**Academic Year:** [Academic Year]

