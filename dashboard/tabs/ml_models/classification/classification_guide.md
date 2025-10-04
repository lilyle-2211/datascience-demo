## What is Classification?
Classification is a supervised machine learning technique that predicts discrete categorical outcomes or classes. Unlike regression which predicts continuous values, classification assigns data points to predefined categories or labels.

**Key Characteristics:**
- **Supervised Learning**: Uses labeled training data
- **Discrete Output**: Predicts categories, not continuous values
- **Decision Boundaries**: Creates boundaries between different classes
- **Probability Estimation**: Can provide confidence scores for predictions

**Types of Classification:**
- **Binary Classification**: Two classes (Yes/No, Churn/No Churn)
- **Multiclass Classification**: Multiple classes (Low/Medium/High Risk)

---

## Use Case Example: Customer Churn Prediction

### Business Problem
Predict which customers are likely to cancel their subscription in the next 30 days, allowing proactive retention efforts.

### Why Classification?
- **Binary Outcome**: Customer will churn (1) or stay (0)
- **Actionable Insights**: Identify high-risk customers for targeted intervention

### Implementation Approach
```python
# Example model structure
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(features, churn_labels)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
churn_probabilities = model.predict_proba(X_test)[:, 1]  # Probability of churn
```

---

## Data Requirements for Classification

### Essential Data Components

**1. Target Variable (Labels)**
- **Binary Example**: Churned (1), Stayed (0)

**2. Feature Variables (Predictors)**
- **Behavioral**: Usage frequency, feature adoption, support tickets
- **Transactional**: Purchase history, payment methods, spending patterns
- **Engagement**: Login frequency, session duration, content consumption

### Customer Churn Data Example

**Dataset Structure:**
```
customer_id | age | tenure_months | monthly_charges | total_charges | support_tickets | last_login_days | churned
1001        | 34  | 24           | 65.50          | 1572.00      | 2              | 3               | 0
1002        | 45  | 6            | 89.90          | 539.40       | 5              | 45              | 1
1003        | 28  | 48           | 45.20          | 2169.60      | 0              | 1               | 0
```

---
## Classification Evaluation Metrics

| Metric      | What it Measures                      | Formula / Description                                 | How to Interpret                | Example & Interpretation |
|-------------|---------------------------------------|------------------------------------------------------|---------------------------------|-------------------------|
| Accuracy    | Overall correct predictions           | (TP + TN) / (TP + TN + FP + FN)                      | Higher is better; all classes equally important | Accuracy = 80% means 80% of all predictions are correct |
| Precision   | Correct positive predictions          | TP / (TP + FP)                                       | Higher is better; focus on positive predictions | Precision = 70% means 70% of predicted positives are correct |
| Recall      | Correctly identified positives        | TP / (TP + FN)                                       | Higher is better; focus on finding all positives | Recall = 90% means 90% of actual positives are found |
| F1-Score    | Balance of precision & recall         | 2 × (Precision × Recall) / (Precision + Recall)       | Higher is better; balance between precision and recall | F1 = 75% means good balance between precision and recall |
| Specificity | Correctly identified negatives        | TN / (TN + FP)                                       | Higher is better; focus on true negatives | Specificity = 85% means 85% of actual negatives are correctly identified |
| AUC-ROC     | Distinguish between classes at all thresholds | Area under ROC curve                        | Closer to 1 is better; model comparison | AUC-ROC = 0.90 means 90% chance model ranks a random positive higher than a random negative |
| AUC-PR      | Precision vs Recall trade-off for rare events | Area under Precision-Recall curve           | Closer to 1 is better; imbalanced datasets | AUC-PR = 0.60 means average precision is 60% across recall levels |

**Notes:**
- TP: True Positives, TN: True Negatives, FP: False Positives, FN: False Negatives
- AUC-PR is more informative than AUC-ROC for imbalanced datasets.
- Always consider business context when choosing metrics.



