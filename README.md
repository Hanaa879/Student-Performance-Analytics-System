# Student-Performance-Analytics-System

An interactive, multi-model Machine Learning analytics suite and dashboard built in Google Colab. This repository contains a modular script that uses the classic Students Performance dataset to perform regression, classification, clustering, dimensionality reduction, and nearest-neighbor peer retrieval across seven different ML models.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [System Architecture](#system-architecture)
4. [Code Explanation](#code-explanation)
5. [Usage Instructions](#usage-instructions)

---

## 1. Overview

The Student Performance Analytics System is designed to explore educational metrics through various machine learning lenses. By utilizing a modular pipeline inside Google Colab, users can dynamically switch between seven models — Linear Regression, Logistic Regression, KNN, Decision Tree, Random Forest, K-Means, and PCA — using an interactive widget dropdown menu without needing to leave the notebook interface.

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
                                 │
       ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼          ▼          ▼
  Linear     Logistic      KNN      Decision    Random     K-Means      PCA
Regression  Regression  (Peer     Tree       Forest    (Clustering) (Dimension-
(Marks       (Pass/    Retrieval) (Perform-  (Dropout   (Student     ality
 Prediction)  Fail)               ance        Risk)      Grouping)   Reduction)
                                   Drivers)
```

---

## 4. Code Explanation

### Part 1: Imports and Environment Setup

The notebook opens by loading the core toolchain needed across every branch of the dashboard, rather than importing things on demand. **Pandas** handles all tabular data operations — reading the CSV, engineering new columns, and dropping/selecting features. **NumPy** is used specifically for `np.where`, which powers the vectorized pass/fail labeling logic, and later for array reshaping in the KNN branch. **Matplotlib** provides the plotting backend for every chart in the notebook, from scatter plots to bar charts to scree plots.

From **scikit-learn**, the imports are organized by role: `train_test_split` is used identically across the regression and classification branches to keep evaluation consistent and comparable; `DecisionTreeClassifier` and `plot_tree` are imported together because the tree model's real value here is both its predictions and its visual interpretability; `classification_report` and `accuracy_score` give a standard evaluation surface shared by every classification branch (Decision Tree, Logistic Regression, Random Forest); and `LabelEncoder`/`StandardScaler` handle the two different preprocessing needs — encoding text into numbers, and rescaling numeric ranges so no single feature dominates distance-based calculations.

Additional model-specific imports (`LinearRegression`, `LogisticRegression`, `RandomForestClassifier`, `KMeans`, `PCA`, `NearestNeighbors`, `ColumnTransformer`, `OneHotEncoder`, `seaborn`) are deliberately imported *inside* their respective branches rather than up front. This keeps the global setup cell lightweight and makes each branch's dependencies self-contained and easy to read in isolation.

Finally, `drive.mount('/content/drive')` authenticates the Colab session against the user's Google account and mounts their Drive as a local filesystem path (`/content/drive`), which is what makes the hardcoded CSV path in Part 2 resolvable at runtime.

### Part 2: Data Loading and Preprocessing

This section transforms the raw CSV into a fully model-ready dataset, and it does so in a deliberate order: **target engineering first, then feature isolation, then encoding, then scaling.**

The dataset is read directly from the mounted Drive path into a single DataFrame called `data`. Two derived target columns are then engineered from the three raw score columns:

- `Average_Score` is computed as the row-wise mean of math, reading, and writing scores. This becomes the continuous target used for regression tasks, since it preserves the full granularity of academic performance.
- `Performance` is derived from `Average_Score` using a 60-point threshold via `np.where`, collapsing the continuous score into a binary label — `1` for students averaging 60 or above (pass), `0` otherwise (fail). This becomes the target for every classification-style task (Logistic Regression, Decision Tree, Random Forest, and the color-coding used in the PCA scatter plot).

Both targets are split off into their own variables (`y_reg` for regression, `y` for classification) before the feature matrix is built, which matters because it prevents any target leakage into the model inputs.

The feature matrix itself (`X_raw`) is created by dropping the three raw score columns *and* both engineered targets from `data` — what remains is purely demographic and categorical information (gender, race/ethnicity, parental education, lunch type, test preparation). Since machine learning models require numeric input, `pd.get_dummies(..., drop_first=True)` One-Hot Encodes these categorical columns into binary indicator columns. Dropping the first category of each feature avoids the "dummy variable trap" — perfect multicollinearity that would otherwise arise from redundant encoding.

Lastly, `StandardScaler` is fit and applied to produce `X_scaled`, which centers each feature around a mean of 0 with unit variance. This step exists specifically for the two unsupervised, distance-sensitive branches — **K-Means** (which groups students based on Euclidean distance between feature vectors) and **PCA** (which finds directions of maximum variance, and is skewed by unscaled features) — since both would otherwise let a feature with a larger raw range dominate the result.

### Part 3: Model Dashboard Controller

The controller is built around a single Colab form parameter, `Select_Model`, which renders as a dropdown widget in the notebook UI (via the `# @param` annotation). Changing this value and re-running the controller cell re-executes only the branch matching the current selection — all branches share the same preprocessed `X`, `y`, `y_reg`, and `X_scaled` from Part 2, so switching models doesn't require re-running the preprocessing step.

**Branch A — Decision Tree (Performance Drivers):**
The encoded feature matrix `X` and binary target `y` are split 80/20 into train and test sets with a fixed `random_state` for reproducibility. A `DecisionTreeClassifier` is instantiated with `max_depth=3` — a deliberate constraint that keeps the tree shallow and interpretable while reducing the risk of overfitting to the training data. After fitting, the model predicts on the held-out test set, and the branch reports both an accuracy score and a full precision/recall/F1 classification report. It also renders the tree structure visually via `plot_tree`, making the decision logic (which feature splits drive a pass/fail prediction) directly inspectable, and separately ranks feature importances to show which inputs the tree relied on most.

**Branch B — Linear Regression (Marks Prediction):**
Here the split uses the continuous target `y_reg` instead of the binary one, since regression predicts an exact average score rather than a class label. A standard `LinearRegression` model is fit on the training split, and predictions are evaluated using three complementary error metrics: Mean Squared Error (penalizes larger errors more heavily), Root Mean Squared Error (same units as the original score, easier to interpret), and R² (how much variance in scores the model explains). The branch finishes with a scatter plot of actual vs. predicted scores overlaid with a baseline trend line — points close to that line indicate accurate predictions, while spread away from it reveals where the model struggles.

**Branch C — Logistic Regression (Pass/Fail Prediction):**
This branch mirrors the Decision Tree's classification setup (same binary target, same split logic) but swaps in `LogisticRegression` with `max_iter=1000` — a higher iteration cap than the default, added because the solver may need more steps to converge on this one-hot encoded feature set. Beyond the standard accuracy/classification report, this branch specifically adds a confusion matrix rendered as a Seaborn heatmap, which visually breaks down true positives, true negatives, false positives, and false negatives — more diagnostic than a single accuracy number, since it shows exactly what kind of errors the model makes.

**Branch D — K-Nearest Neighbors (Peer Retrieval):**
This branch diverges from the others structurally, since KNN here is used for peer retrieval rather than prediction. A `ColumnTransformer` is set up to preprocess numeric score columns (via `StandardScaler`) and categorical demographic columns (via `OneHotEncoder`) through separate pipelines simultaneously, ensuring each feature type is treated appropriately for distance calculations — this is a fresh, KNN-specific encoding rather than reusing `X_scaled` from Part 2. A `NearestNeighbors` model is then fit using Euclidean distance with `n_neighbors=6` — retrieving five peers plus the target student itself. The branch's real work is in its custom dashboard function: it renders the selected target student's profile as an HTML summary card, plots each retrieved peer's distance from the target as a horizontal bar chart (shorter bars = more similar students), and compares the target's individual scores against the average scores of their retrieved peer group — surfacing whether a student is over- or under-performing relative to demographically similar peers.

**Branch E — Random Forest (Dropout Risk):**
Structurally close to the Decision Tree branch — same binary target, same 80/20 split — but replaces the single tree with a `RandomForestClassifier` made up of `n_estimators=100` individual trees, each trained on a bootstrapped sample with random feature subsets. This ensemble approach typically generalizes better than one tree, since it averages out the idiosyncrasies of any single tree's splits. After reporting accuracy and a classification report, the branch ranks and displays the top 10 most important features (aggregated across all 100 trees this time, rather than one), and renders them as a horizontal bar chart — giving a more stable, less overfit view of which factors actually drive outcomes compared to the single Decision Tree's importances.

**Branch F — K-Means (Student Grouping & Segmentation):**
This is the first fully unsupervised branch — it uses no target variable at all, only the scaled feature matrix `X_scaled` from Part 2. `KMeans` is configured with `n_clusters=3` (looking for three natural groupings) and `n_init=10` (running the algorithm 10 times with different centroid seeds and keeping the best result, since K-Means can converge to different local optima depending on initialization). The resulting cluster labels are written back onto the original, human-readable `data` DataFrame as a new `Cluster_Group` column. The branch then computes each cluster's mean math/reading/writing/average scores to build a profile of "who" each discovered group actually is (e.g., a high performers group vs. a struggling group), visualizes this with a grouped bar chart, and prints a count of how many students fall into each segment.

**Branch G — PCA (Feature Reduction & Visualization):**
Also unsupervised and also built on `X_scaled`. Two PCA models are fit: a full-dimensionality `pca_full` (used only to inspect how variance is distributed across *all* possible components) and a 2-component `pca` used for the actual visualization and `X_pca` projection. The branch first prints how much of the dataset's total variance is captured by each of the two retained components individually, plus their combined total — this quantifies how much information is "lost" by compressing everything down to two dimensions. It then produces two plots: a 2D scatter of every student projected onto Principal Component 1 vs. Principal Component 2, color-coded by the binary `Performance` target (`y`) so any natural separation between passing and failing students becomes visible; and a scree plot — a bar chart of variance explained per component across the *full* set — which is the standard way to judge how many components would be "worth" keeping if you were doing further dimensionality reduction.

---

## 5. Usage Instructions

1. Open your notebook inside Google Colab.
2. Upload the `StudentsPerformance.csv` file into your Google Drive under the path `/content/drive/MyDrive/archive (2)/StudentsPerformance.csv`.
3. Execute the setup, import, and preprocessing code blocks sequentially.
4. Select your preferred analytical model — Linear Regression, Logistic Regression, KNN, Decision Tree, Random Forest, K-Means, or PCA — from the `Select_Model` dropdown parameter and run the controller block to render the corresponding analytics dashboard and visualizations.
