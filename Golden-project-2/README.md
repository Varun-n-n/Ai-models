# Personality Prediction System

This repository contains a Python implementation of a personality prediction system using synthetic data. The system predicts personality traits based on image features using machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results](#Results)
- [Author](#Author)

## Introduction

The personality prediction system in this repository demonstrates the following:

- Generation of a synthetic dataset for personality prediction.
- Splitting the dataset into training and testing sets.
- Training a Random Forest regressor for personality prediction.
- Evaluating the model's performance using Root Mean Squared Error (RMSE).
- Visualizing the predicted vs. actual personality trait values with improved scatter plots.

## Dataset

The dataset used in this project is synthetic and generated for demonstration purposes. It includes random image features and personality trait scores.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/Varun-n-n/Ai-models.git
    cd Golden-project-2
    ```

2. Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the personality prediction script:

    ```bash
    python Personality-prediction-Via_CV_analysis.py
    ```

4. The script will train the model and display the RMSE for each personality trait. It will also generate scatter plots for visualization.

## Requirements

- Python 3.x
- NumPy
- pandas
- scikit-learn
- matplotlib

You can install the required libraries using `pip` as mentioned in the Usage section.

## Author

- HEMANTH KUMAR KS(https://github.com/Varun-n-n)
