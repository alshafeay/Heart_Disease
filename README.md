# Heart_Disease
# Comprehensive Machine Learning Pipeline for Heart Disease Prediction

## 1. Project Overview

This project implements a full-stack machine learning pipeline to analyze, predict, and visualize heart disease risk using the UCI Heart Disease dataset. The workflow covers everything from data preprocessing and exploratory data analysis to model training, hyperparameter tuning, and deployment of an interactive web application.

The primary goal is to build a reliable classification model and embed it into a user-friendly interface, demonstrating a complete end-to-end data science project.

---

## 2. Key Features

- **Data Cleaning:** Handles missing values and prepares the dataset for analysis.
- **Exploratory Data Analysis (EDA):** Visualizes data distributions and feature correlations.
- **Feature Engineering:** Includes dimensionality reduction with PCA and feature selection with Chi-Square, Random Forest Importance, and RFE.
- **Supervised Learning:** Trains and evaluates four classification models:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
- **Unsupervised Learning:** Explores natural data clusters using K-Means and Hierarchical Clustering.
- **Model Optimization:** Enhances the best-performing model using `GridSearchCV` for hyperparameter tuning.
- **Interactive UI:** A web application built with Streamlit allows users to input their health data and receive real-time predictions.
- **Deployment:** The local web application is made publicly accessible using Ngrok.

---

## 3. File Structure
```
Heart_Disease_Project/
│
├── data/
│   ├── heart_disease.csv
│   ├── heart_disease_cleaned.csv
│   ├── heart_disease_pca.csv
│   └── heart_disease_selected_features.csv
│
├── deployment/
│   └── ngrok_setup.txt
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── results/
│   └── evaluation_metrics.md
│
├── ui/
│   └── app.py
│
├── .gitignore
├── README.md
└── requirements.txt
---
```
## 4. Setup and Installation

To run this project locally, please follow these steps:

**Prerequisites:**
- Python 3.8 or higher
- Pip (Python package installer)

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Heart_Disease_Project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 5. How to Use the Project

### Running the Analysis Scripts

You can run the analysis steps sequentially using the Python scripts or Jupyter Notebooks provided. For example, to run the data preprocessing script:
```bash
python 01_data_preprocessing.py
Running the Web Application
Start the Streamlit App:
Make sure you are in the root Heart_Disease_Project directory.

Bash

streamlit run ui/app.py
This will open the application in your local browser.

Deploy with Ngrok (Optional):
To share the app, open a second terminal and run:

Bash

ngrok http 8501
Use the public URL provided by Ngrok to access the app from anywhere.

6. Model Summary
The final model selected for deployment was a Tuned Random Forest Classifier. After hyperparameter optimization, it achieved strong performance metrics, proving to be the most robust model for this classification task. For a detailed breakdown of all model results, please see the results/evaluation_metrics.md file.

7. Tools and Libraries
Python

Pandas & NumPy: For data manipulation.

Matplotlib & Seaborn: For data visualization.

Scikit-learn: For machine learning modeling and preprocessing.

Streamlit: For building the interactive web UI.

Ngrok: For deploying the local web application.

Jupyter Notebook: For exploratory analysis.
