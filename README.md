# 🧠 MLProject-Early_Stroke_Risk_Prediction_System

## 📌 Project Overview
**EarlyStrokeRiskPrediction-RandomForest** is a machine learning project designed to predict whether a patient is at risk of having a **stroke** based on health and lifestyle attributes. The project uses the **Healthcare Stroke Dataset** and applies a **Random Forest Classifier** to perform binary classification.

The system demonstrates a complete machine learning workflow including **data preprocessing, class imbalance handling, encoding, model training, prediction, and evaluation**.

---

## 🎯 Objectives
- Predict stroke risk using patient health and lifestyle metrics.
- Apply **Random Forest classification** for medical data analysis.
- Handle **class imbalance** using SMOTE oversampling.
- Demonstrate encoding of categorical features and model evaluation techniques.
- Provide insights into feature importance for stroke prediction.

---

## 🗂️ Dataset Description
The project uses the **Healthcare Dataset – Stroke Data**, a medical dataset for stroke risk prediction tasks.

### Key Features
| Column | Description |
|--------|-------------|
| `gender` | Gender of the patient |
| `age` | Age of the patient |
| `hypertension` | 0 → No hypertension, 1 → Has hypertension |
| `heart_disease` | 0 → No heart disease, 1 → Has heart disease |
| `ever_married` | Marital status of the patient |
| `work_type` | Type of occupation |
| `Residence_type` | Urban or Rural residence |
| `avg_glucose_level` | Average blood glucose level |
| `bmi` | Body Mass Index |
| `smoking_status` | Smoking history of the patient |

### Target Variable
| Column | Description |
|--------|-------------|
| `stroke` | 0 → No Stroke, 1 → Stroke |

---

## 🔧 Tools & Technologies
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **Jupyter Notebook / Google Colab**

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Loaded the dataset and performed **Exploratory Data Analysis (EDA)**.
- Dropped the irrelevant `id` column.
- Handled missing values in the `bmi` column by replacing `NaN` with the **median value**.
- Generated a **Correlation Heatmap** for numerical columns.

### 2️⃣ Data Preparation
- Identified **categorical columns**: `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`.
- Applied **OneHotEncoder via ColumnTransformer** to encode categorical features.
- Created **feature matrix (X)** and **target variable (y)**.
- Split the dataset into **training set and testing set** (80/20 split with stratification).

### 3️⃣ Handling Class Imbalance
- The stroke dataset is highly imbalanced (very few stroke cases).
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** on the training set to balance the class distribution before model training.

### 4️⃣ Feature Scaling
- Applied **StandardScaler** to normalize features after SMOTE.

### 5️⃣ Model Training
- Applied **Random Forest Classifier** with `n_estimators=500` for stroke prediction.
- Trained the model on the **SMOTE-balanced training dataset**.
- Generated predictions for the **test dataset**.

### 6️⃣ Model Evaluation
The performance of the model was evaluated using multiple classification metrics:

#### Confusion Matrix
```
[[962  10]
 [ 48   2]]
```

#### Metrics at Default Threshold
| Metric | Score |
|--------|-------|
| Accuracy | 0.9432 |
| Sensitivity (Recall) | 0.16 |
| Specificity | 0.9650 |
| ROC-AUC Score | 0.7837 |

#### Threshold Tuning Results
| Threshold | Sensitivity | Specificity |
|-----------|-------------|-------------|
| 0.4 | 0.14 | 0.98 |
| 0.3 | 0.16 | 0.97 |
| 0.2 | 0.20 | 0.91 |
| 0.1 | 0.54 | 0.82 |

> ⚠️ Due to the highly imbalanced nature of stroke data, sensitivity is intentionally tuned via threshold adjustment to improve detection of true stroke cases at the cost of some specificity.

### 7️⃣ Feature Importance
Top contributing features identified by the Random Forest model:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Age | 0.2075 |
| 2 | Hypertension | 0.0855 |
| 3 | Ever Married (Yes) | 0.0649 |
| 4 | Avg Glucose Level | 0.0602 |
| 5 | Heart Disease | 0.0550 |

---

## 📊 Sample Outputs
- Correlation Heatmap
- SMOTE Class Distribution Before & After
- Confusion Matrix
- Accuracy Score
- Sensitivity and Specificity Calculation
- Threshold Tuning Analysis
- Feature Importance Barplot
- ROC-AUC Score

---

## 🚀 Potential Applications
- 🏥 Early stroke risk screening in clinical settings
- 📊 Healthcare decision support systems
- 🧠 Clinical data analysis and pattern recognition
- 📈 Medical machine learning research

---

## ▶️ How to Run the Project

### Run the Model in Google Colab (Recommended for Quick Start)

### Step 1: Upload notebook to Google Colab
### Step 2: Upload `healthcare-dataset-stroke-data.csv` to the Colab environment
### Step 3: Run all cells
