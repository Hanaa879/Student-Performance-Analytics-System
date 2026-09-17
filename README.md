# Student-Performance-Analytics-System

An interactive, multi-model Machine Learning analytics suite and dashboard built in Google Colab. This repository contains a modular script that uses the classic Students Performance dataset to perform regression, classification, clustering, and nearest-neighbor peer retrieval.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [System Architecture](#system-architecture)
4. [Code Explanation](#code-explanation)
5. [Usage Instructions](#usage-instructions)

---

## 1. Overview

The Student Performance Analytics System is designed to explore educational metrics through various machine learning lenses. By utilizing a modular pipeline inside Google Colab, users can dynamically switch between regression, classification, and retrieval tasks using an interactive widget dropdown menu without needing to leave the notebook interface.

---

## 2. Dataset Structure

The system expects the `StudentsPerformance.csv` dataset, which contains the following features:

* **Demographics:** Gender, Race/Ethnicity, Parental level of education
* **Socioeconomic/Prep Factors:** Lunch type (standard vs. free/reduced), Test preparation course (none vs. completed)
* **Academic Scores:** Math score, Reading score, Writing score (Numeric values from 0 to 100)

---

## 3. System Architecture

```text
[ Google Drive CSV ] 
       │
       ▼
[ Data Loading & Preprocessing ] 
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
[ Feature Engineering ]   [ StandardScaler ]        [ One-Hot Encoding ]
(Average Score, Targets)  (Distance-based models)   (Categorical conversion)
       │
       └─────────────────────────┬─────────────────────────┘
                                 ▼
                 [ Interactive Model Dashboard Selector ]
```

---

## 4. Code Explanation

### Part 1: Imports and Environment Setup

The script begins by importing the core data science stack — **Pandas** and **NumPy** for data manipulation and numerical operations, and **Matplotlib** for visualizations. From **scikit-learn**, it pulls in `train_test_split` for splitting data, `DecisionTreeClassifier` and `plot_tree` for the decision tree model and its visualization, `classification_report` and `accuracy_score` for evaluating classifiers, and `LabelEncoder`/`StandardScaler` for preparing categorical and numerical features. Finally, it mounts Google Drive so the notebook can read the dataset directly from cloud storage.

### Part 2: Data Loading and Preprocessing

The dataset is loaded from its Google Drive path into a DataFrame. Two target variables are then engineered: an `Average_Score` column, computed as the row-wise mean of the math, reading, and writing scores, and a binary `Performance` column that labels a student as passing (`1`) if their average is 60 or above, and failing (`0`) otherwise. The raw score columns and both targets are dropped from the feature set, leaving only demographic and categorical inputs, which are then One-Hot Encoded to convert text categories into numeric columns. Finally, a `StandardScaler` is applied to normalize these features for distance-sensitive algorithms like KNN.

### Part 3: Model Dashboard Controller

An interactive Colab form parameter (`Select_Model`) lets the user pick a model from a dropdown — Linear Regression, Logistic Regression, KNN, Decision Tree, Random Forest, K-Means, or PCA — and the script branches into the corresponding block based on that selection.

**Branch A — Decision Tree:** Splits the encoded data into train/test sets, fits a `DecisionTreeClassifier` with a max depth of 3 to control overfitting, and generates predictions. Outputs include accuracy score, a full classification report, a visual decision tree diagram, and feature importance rankings.

**Branch B — Linear Regression:** Splits the data using the continuous `Average_Score` target, trains a `LinearRegression` model, and evaluates it with MSE, RMSE, and R² metrics. Produces a scatter plot comparing actual vs. predicted scores against a baseline trend line.

**Branch C — Logistic Regression:** Splits the data using the binary `Performance` target, trains a `LogisticRegression` classifier (with a higher iteration cap for convergence), and evaluates it with standard classification metrics plus a Seaborn confusion matrix heatmap.

**Branch D — K-Nearest Neighbors (KNN):** Builds a `ColumnTransformer` to handle numeric and categorical features together, then fits a `NearestNeighbors` model using Euclidean distance to find each student's closest peers. A custom dashboard function renders the target student's profile as an HTML card, plots peer distance/closeness as horizontal bar charts, and compares the target's scores against their peer group's average.

---

## 5. Usage Instructions

1. Open your notebook inside Google Colab.
2. Upload the `StudentsPerformance.csv` file into your Google Drive under the path `/content/drive/MyDrive/archive (2)/StudentsPerformance.csv`.
3. Execute the setup, import, and preprocessing code blocks sequentially.
4. Select your preferred analytical model from the `Select_Model` dropdown parameter and run the controller block to render the corresponding analytics dashboard and visualizations.
