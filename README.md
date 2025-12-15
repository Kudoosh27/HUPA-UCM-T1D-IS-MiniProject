# Intelligent Glucose Level Prediction System for Type 1 Diabetes

This repository contains the implementation of an intelligent system for predicting blood glucose levels in patients with Type 1 Diabetes using machine learning and deep learning techniques.

## Dataset
The dataset used in this study is `cleaned_all_participants.csv`, which contains continuous glucose monitoring (CGM) data, insulin delivery information, dietary intake, and wearable sensor measurements.

## Models Implemented
- Artificial Neural Network (ANN)
- Long Short-Term Memory (LSTM)

## Preprocessing
All preprocessing steps, including normalization, sequence generation, and train-test splitting, are performed programmatically to ensure reproducibility.

## Usage
1. Place `cleaned_all_participants.csv` inside the `dataset/` folder.
2. Run `preprocessing.py`
3. Run `ann_model.py` or `lstm_model.py`
