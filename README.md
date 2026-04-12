# Student Performance ML Replication

This repository contains code for replicating a machine learning study on predicting secondary school student performance.

## Paper
Cortez & Silva (2008) – Using Data Mining to Predict Secondary School Student Performance.

## Dataset
UCI Student Performance Dataset:

https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set

https://archive.ics.uci.edu/ml/datasets/student+performance

This project uses the Mathematics dataset (student-mat.csv).

## How to run

1. Install dependencies:
pip install -r requirements.txt

2. Run the script:
python replicate_student_performance.py

The script will:
- load the dataset
- train Decision Tree and Random Forest models
- output evaluation metrics

## Notes
- The replication focuses on binary classification (pass/fail).
- Pass = G3 >= 10
