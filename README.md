# Student Performance ML Replication

This repository contains code for replicating a machine learning study on predicting secondary school student performance.

## Paper

Cortez & Silva (2008) – *Using Data Mining to Predict Secondary School Student Performance*.

## Dataset

Primary source (Kaggle):
https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set

Original source (UCI):
https://archive.ics.uci.edu/ml/datasets/student+performance

The dataset (`student-mat.csv`) is included in this repository for reproducibility.

## Requirements

* pandas
* scikit-learn

## How to Run

### Option 1: Google Colab

1. Upload all files (`replicate_student_performance.py`, `student-mat.csv`, and `requirements.txt`) to Colab.
2. Run the following in a code cell:

```python
!pip install -r requirements.txt
!python replicate_student_performance.py
```

> Note: In Google Colab, commands such as `pip` and `python` must be prefixed with `!` because notebook cells run Python by default.

### Option 2: Local Environment

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the script:

```bash
python replicate_student_performance.py
```

## What the Script Does

* Loads the dataset
* Creates a binary target variable (`pass` / `fail`)
* Trains Decision Tree and Random Forest models
* Prints evaluation metrics for both experiments

## Notes

* The replication focuses on binary classification (pass/fail).
* A student is considered to have passed if `G3 >= 10`.

