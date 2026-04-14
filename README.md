# Student Performance ML Replication

This repository contains code for replicating a machine learning study on predicting secondary school student performance.

## Paper
Cortez & Silva (2008) – Using Data Mining to Predict Secondary School Student Performance.

## Dataset
Primary source (Kaggle):  
https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set

Original source (UCI):  
https://archive.ics.uci.edu/ml/datasets/student+performance

The dataset (`student-mat.csv`) is included in this repository for reproducibility.

## Requirements
- pandas
- scikit-learn

## How to Run

### Option 1: Google Colab (Recommended)

1. Upload all files (`.py`, `.csv`, `requirements.txt`) to Colab.
2. Run the following in a code cell:

```python
!pip install -r requirements.txt
!python replicate_student_performance.py
```

---

### Option 2: Local Environment

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the script:

```
python replicate_student_performance.py
```

---

### What the script does

* Loads the dataset
* Creates a binary target (pass/fail)
* Trains Decision Tree and Random Forest models
* Prints evaluation metrics


## Notes
- The replication focuses on binary classification (pass/fail).
- Pass = G3 >= 10
