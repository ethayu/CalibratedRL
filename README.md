# CalibratedRL: Distribution Calibration for Reinforcement Learning

This repository contains the official implementation for the paper **"Distribution Calibration for Reinforcement Learning"**. The paper explores the importance of calibrated uncertainties in predictive models and introduces simple yet effective methods to improve performance using **Isotonic Regression** and **GP-Beta** calibration.

---

## 📄 [Paper](paper.pdf) Abstract

Estimates of predictive uncertainty are crucial for accurate model-based planning and reinforcement learning. However, predictive uncertainties—especially those derived from modern deep learning systems—are often inaccurate, limiting performance.

This [paper](paper.pdf) argues that good uncertainties must be calibrated, ensuring that predicted probabilities match empirical frequencies of events. We describe straightforward approaches to augment any model-based reinforcement learning agent with calibrated models. Using **Isotonic Regression** and **GP-Beta** for calibration, we demonstrate consistent improvements in:

- Planning
- Sample complexity
- Exploration

---

## 🚀 Getting Started

Follow the steps below to reproduce the experiments and results from the paper.

### 1. Prerequisites

Ensure you have Python installed (version 3.6 or higher). Install the required libraries:

```bash
pip install torch numpy pandas netcal
```

### 2. Dataset Preparation

Download the **Corporación Favorita Grocery Sales** dataset from [Kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting) and rename it as the `train.csv` file in the `data/` directory.
```bash
python -m data.data_processer
```

This will generate `train_processed.csv` and `test_processed.csv` in the `data/` directory.

### 4. Training the Models

#### 4.1 Train the Transition Model

Train the Bayesian neural network transition models:
```bash
python -m training.train_dynamics
```

This script trains the transition model for each item in the dataset.

#### 4.2 Train the Calibration Models

Train the calibration models (Isotonic Regression and GP-Beta):
```bash
python -m training.train_calibration
```
This script fits calibration models to the outputs of the transition models to improve uncertainty estimates.

### 5. Evaluation

Run the evaluation script to assess the performance of the planners with calibrated and uncalibrated models:
```bash
python -m training.evaluate
```

This will execute rollouts using both the heuristic and MPC planners, with and without calibration, and record the results.

### 6. Aggregating Results

Since evaluation can take a long time, you may want to look at intermediate results or analyze the outputs at any point. Use the following command to aggregate and summarize the results:

```bash
python -m data.aggregate_results
```

This script collects results from the evaluation runs and generates summary statistics and visualizations.

---

## 📊 Experimental Results

Our experiments demonstrate that calibrating transition models significantly improves planning outcomes in model-based reinforcement learning. Key findings include:

- **Reduced Waste**: Calibration methods decreased inventory waste by up to 15%.
- **Increased Rewards**: Overall rewards increased by 20% with calibrated models.
- **Enhanced Reliability**: Calibrated models provided more accurate uncertainty estimates, leading to better decision-making.

---

## 📝 Repository Structure

- `data/`: Contains data processing scripts and the processed datasets.
- `training/`: Includes scripts for training transition and calibration models.
- `models/`: Directory where trained model weights are saved.
- `evaluation/`: Contains evaluation scripts and utilities.
- `results/`: Stores output from evaluations and aggregated results.

---

## 🔧 Implementation Details

### Transition Model

- **Architecture**: Bayesian neural network with five hidden layers, each containing 128 units and ReLU activations.
- **Uncertainty Estimation**: Uses Monte Carlo Dropout with a rate of 0.5 to generate probabilistic predictions.
- **Training**: Trained individually for each item in the dataset.

### Calibration Methods

#### Isotonic Regression

- **Purpose**: Non-parametric calibration method that adjusts the predicted cumulative distribution function (CDF) to better align with empirical data.
- **Implementation**: Utilizes the `netcal` library for fitting the isotonic regression model.

#### GP-Beta

- **Purpose**: Applies Gaussian Process regression to estimate parameters of a Beta calibration function, refining the predicted CDF.
- **Implementation**: Also implemented using the `netcal` library.

---