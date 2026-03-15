# Titanic Survival Prediction: Machine Learning from Disaster

## 1. Project Overview & Dataset Description

### The Problem

The goal of this project is to predict whether a passenger survived the sinking of the RMS Titanic. On April 15, 1912, the Titanic sank after colliding with an iceberg, resulting in the death of 1502 out of 2224 passengers and crew. While survival involved luck, certain groups of people (such as women, children, and the upper-class) were more likely to survive than others.

### The Dataset

The project uses the classic Kaggle Titanic dataset, which is split into a training set (`train.csv`) and a test set (`test.csv`).

**Data Types included:**

* **Numerical:** `Age` (continuous), `Fare` (continuous), `SibSp` (count of siblings/spouses), `Parch` (count of parents/children).
* **Categorical:** `Survived` (Target: 0 or 1), `Pclass` (Socio-economic status: 1, 2, or 3), `Sex` (male, female), `Embarked` (Port of embarkation).
* **Text/Mixed:** `Name` (contains titles), `Ticket` (alphanumeric), `Cabin` (deck information).

---

## 2. The Core Challenge: Feature Engineering vs. Modeling

In this project, **Feature Engineering is more important than Modeling.**

**Why?**
The Titanic dataset is relatively small (891 training samples). In small datasets, complex models like Deep Neural Networks or heavy Gradient Boosting often overfit the noise rather than learning general patterns. The "signal" in this data is hidden in the raw text and relationships:

* A model cannot naturally understand that "Mrs." implies a specific social status.
* A model doesn't know that the first letter of a `Cabin` number (the Deck) correlates with the distance to the lifeboats.

By manually extracting these insights, we simplify the problem so that even a basic model (like Logistic Regression) can achieve high accuracy. Good features define the "ceiling" of your performance; the model simply helps you reach it.

---

## 3. Feature Engineering Steps

Our pipeline focuses on transforming raw data into meaningful numeric "signals" while preventing data leakage:

1. **Title Extraction:** We extract titles (Mr, Mrs, Miss, Master, etc.) from the `Name` column. Rare titles (like Dr, Rev, or Lady) are consolidated into a "Rare" category to reduce noise.
2. **Missing Value Imputation:**
* **Age:** Instead of a simple average, we use **KNN Imputation** based on `Pclass` and `Title` to provide a more logical age guess.
* **Fare & Embarked:** Missing values are filled using the median (per class) or the most frequent value (mode).


3. **Binning:** Continuous variables like `Age` and `Fare` are converted into categorical "groups" (e.g., Child, Teen, Adult) to help models handle non-linear relationships.
4. **Cabin Processing:** We extract the **Deck** letter from the `Cabin` column, as the deck level was a major factor in how quickly a passenger could reach the top deck.
5. **Family Dynamics:** We combine `SibSp` and `Parch` to calculate `FamilySize` and create a boolean `IsAlone` feature.
6. **Encoding:** Binary variables (Sex) are mapped to 0/1, and multi-category variables (Embarked, Title, Deck) are transformed using **One-Hot Encoding**.

---

## 4. Modeling Steps

Once the data is cleaned and engineered, we follow a rigorous training process:

1. **Baseline Testing:** We test multiple algorithms, including **Logistic Regression, Random Forest, Gradient Boosting, and SVM**.
2. **Pipeline Integration:** To prevent data leakage, we use Scikit-Learn `Pipelines` that include `MinMaxScaler` or `StandardScaler`. This ensures that scaling parameters are only learned from the training data.
3. **Hyperparameter Tuning:** We use **GridSearchCV** to find the optimal settings (like the number of trees in a forest or the "C" penalty in Logistic Regression) for each model.
4. **Cross-Validation:** We use **10-Fold Stratified Cross-Validation**. This means the model is trained and tested 10 different times on different slices of data to ensure the accuracy score is stable and not a fluke.
5. **Ensembling (Voting Classifier):** We combine the best-performing models into a **Soft-Voting Ensemble**. By letting different models "vote" on the outcome, we cancel out individual model errors and create a more robust final prediction.
6. **Validation:** Final performance is checked against a local validation set to ensure the model generalizes well to unseen passengers before submitting to Kaggle.