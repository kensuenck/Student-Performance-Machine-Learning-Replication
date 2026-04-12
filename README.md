# Student Performance ML Replication

This repository contains code for replicating a machine learning study on predicting secondary school student performance.

## Paper
Cortez & Silva (2008) – Using Data Mining to Predict Secondary School Student Performance.

## Dataset
Primary source (Kaggle):
https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set

Original source (UCI):
https://archive.ics.uci.edu/ml/datasets/student+performance

The dataset (student-mat.csv) is included in this repository for reproducibility.

## How to run

## Reproducibility

This repository is designed to run in a single execution.

All required files, including the dataset, are included.

Steps:
1. Install dependencies:
pip install -r requirements.txt

2. Run:
python replicate_student_performance.py

The script will:
- load the dataset
- train Decision Tree and Random Forest models
- output evaluation metrics

## Notes
- The replication focuses on binary classification (pass/fail).
- Pass = G3 >= 10
