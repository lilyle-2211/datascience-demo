## What is Regression?
Regression is a supervised machine learning technique that predicts continuous numeric outcomes. Unlike classification, which predicts discrete categories, regression estimates a value (e.g., price, score, or lifetime value) for each data point.

**Key Characteristics:**
- **Supervised Learning**: Uses labeled training data
- **Continuous Output**: Predicts real-valued numbers
- **Relationship Modeling**: Captures relationships between features and target
- **Error Minimization**: Optimizes for smallest prediction error

---

## Use Case Example: Customer Lifetime Value (LTV) Prediction

### Business Problem
Estimate the future revenue a customer will generate (LTV), enabling better marketing, retention, and resource allocation decisions.

### Why Regression?
- **Continuous Outcome**: LTV is a numeric value
- **Personalized Insights**: Identify high-value customers
- **Business Impact**: Optimize marketing spend and retention strategies

### Implementation Approach
```python
# Example model structure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(features, ltv_values)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
ltv_predictions = model.predict(X_test)
```

---

## Data Requirements for Regression

### Essential Data Components

**1. Target Variable (Labels)**
- **Continuous Example**: Customer LTV, sales amount, price

**2. Feature Variables (Predictors)**
- **Demographic**: Age, gender, location
- **Behavioral**: Usage frequency, feature adoption, support tickets
- **Transactional**: Purchase history, payment methods, spending patterns
- **Engagement**: Login frequency, session duration, content consumption

### Customer LTV Data Example

**Dataset Structure:**
```
customer_id | age | tenure_months | monthly_charges | total_charges | support_tickets | last_login_days | ltv
1001        | 34  | 24           | 65.50          | 1572.00      | 2              | 3               | 1200.00
1002        | 45  | 6            | 89.90          | 539.40       | 5              | 45              | 400.00
1003        | 28  | 48           | 45.20          | 2169.60      | 0              | 1               | 2100.00
```

---

## Regression Evaluation Metrics

| Metric      | What it Measures                      | Formula / Description                                 | How to Interpret                | Example & Interpretation |
|-------------|---------------------------------------|------------------------------------------------------|---------------------------------|-------------------------|
| R² (R-squared) | Variance explained by the model     | 1 - (Sum of Squared Errors / Total Variance)          | Closer to 1 means better fit    | R² = 0.85 means 85% of variance is explained by the model |
| MSE         | Average squared error                 | Mean((Actual - Predicted)^2)                          | Lower is better; in squared units| MSE = 1000 means average squared error is 1000 |
| RMSE        | Root of average squared error         | sqrt(Mean((Actual - Predicted)^2))                    | Lower is better; in original units| RMSE = 30 means average error is about 30 units |
| MAE         | Average absolute error                | Mean(|Actual - Predicted|)                            | Lower is better; in original units| MAE = 20 means average error is 20 units |
| MAPE        | Average absolute percentage error     | Mean(|Actual - Predicted| / |Actual|) × 100%          | Lower is better; as a percentage | MAPE = 10% means predictions are off by 10% on average |

**Notes:**
- R² close to 1 means good fit; compare train/test for overfitting.
- RMSE and MAE are in the same units as the target variable.
- MAPE is easy to interpret as a percentage, but can be misleading if actual values are near zero.
- Always use multiple metrics and consider business context.
