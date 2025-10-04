# LightGBM Learn-to-Rank Guide

## Overview
LightGBM Learn-to-Rank is a machine learning technique for ranking items based on their relevance to users. Unlike regression (predicting exact values) or classification (predicting categories), ranking optimizes the relative order of items.

## 1. Data Requirements & Format

### Data Structure
Learn-to-rank requires data organized in **query-document pairs**:
- **Query**: Represents a user, search term, or context
- **Document**: Represents an item to be ranked (product, movie, etc.)
- **Relevance**: Score indicating how relevant the document is to the query

### Required Data Format

#### Training Data Structure
```
query_id | document_features | relevance_score
---------|------------------|----------------
user_1   | [f1, f2, f3...]  | 4
user_1   | [f1, f2, f3...]  | 2  
user_1   | [f1, f2, f3...]  | 5
user_2   | [f1, f2, f3...]  | 3
user_2   | [f1, f2, f3...]  | 1
```

#### Essential Components:
1. **Query Groups**: Items grouped by query/user (same query_id)
2. **Features**: Numerical features describing each item
3. **Labels**: Relevance scores (typically 1-5 or 0-4)
4. **Group Information**: Number of items per query group

#### Example Dataset Structure:
```python
# Features (X): Shape (n_samples, n_features)
X = [[feature1, feature2, feature3, ...],  # Item 1 for User A
     [feature1, feature2, feature3, ...],  # Item 2 for User A  
     [feature1, feature2, feature3, ...],  # Item 3 for User A
     [feature1, feature2, feature3, ...],  # Item 1 for User B
     [feature1, feature2, feature3, ...]]  # Item 2 for User B

# Labels (y): Relevance scores
y = [4, 2, 5, 3, 1]  # Corresponding relevance scores

# Groups: Number of items per query
groups = [3, 2]  # User A has 3 items, User B has 2 items
```

### Feature Engineering for Ranking

#### User Features:
- User demographics (age, location)
- User behavior (avg_rating, num_purchases, rating_std)
- User preferences (category preferences)

#### Item Features:
- Item properties (price, category, release_year)
- Item popularity (avg_rating, num_ratings)
- Item age (current_year - release_year)

#### Interaction Features:
- User-item similarity scores
- Historical interaction patterns
- Context-dependent features (time, location)

## 2. Learn-to-Rank Training Concept

### How Training Works

#### Pairwise Learning:
1. **Generate Pairs**: For each query group, create pairs of items
2. **Compare Relevance**: Determine which item in each pair should rank higher
3. **Learn Preferences**: Train model to predict correct pairwise preferences
4. **Optimize Ranking**: Model learns to rank items in correct relative order

#### Training Process:
```
Query Group: User_1
Items: [ItemA(score=5), ItemB(score=2), ItemC(score=4)]

Generated Pairs:
- ItemA vs ItemB: ItemA should rank higher (5 > 2) ✓
- ItemA vs ItemC: ItemA should rank higher (5 > 4) ✓  
- ItemB vs ItemC: ItemC should rank higher (4 > 2) ✓

Model learns: ItemA > ItemC > ItemB
```

#### Key Training Principles:
- **Relative Learning**: Model learns relative preferences, not absolute scores
- **Group-wise Optimization**: Training considers entire ranking lists per query
- **Loss Functions**: Optimize ranking-specific losses (e.g., LambdaRank)

## 3. Evaluation Metrics

### Understanding NDCG (Normalized Discounted Cumulative Gain)

#### What is NDCG?
NDCG is the **gold standard metric for ranking systems**. It measures how well your ranking matches the ideal ranking, with special emphasis on getting the most relevant items at the top positions.

#### Core Concepts:

**1. Cumulative Gain (CG):**
Simply the sum of relevance scores in your ranking.
```
CG@k = relevance₁ + relevance₂ + ... + relevanceₖ
```

**2. Discounted Cumulative Gain (DCG):**
Adds **position discounting** - items lower in the ranking contribute less to the score.
```
DCG@k = Σ(i=1 to k) [relevance_i / log₂(i+1)]
```

**Why discounting?** Users are more likely to see and click items at the top. Position 1 gets full weight, position 2 gets weight 1/log₂(3) ≈ 0.63, position 3 gets 1/log₂(4) = 0.5, etc.

**3. Normalized DCG (NDCG):**
Normalizes DCG by the **ideal DCG** (IDCG) to enable comparison across different query groups.
```
NDCG@k = DCG@k / IDCG@k

Where IDCG@k = DCG of the perfect ranking
```

#### NDCG Step-by-Step Example:

**Scenario:** Recommending 5 products, relevance scores 1-5
```
Your Ranking:    [Product_A(rel=4), Product_B(rel=2), Product_C(rel=5), Product_D(rel=1), Product_E(rel=3)]
Ideal Ranking:   [Product_C(rel=5), Product_A(rel=4), Product_E(rel=3), Product_B(rel=2), Product_D(rel=1)]
```

**Step 1: Calculate DCG@5 for your ranking**
```
DCG@5 = 4/log₂(2) + 2/log₂(3) + 5/log₂(4) + 1/log₂(5) + 3/log₂(6)
      = 4/1.0 + 2/1.58 + 5/2.0 + 1/2.32 + 3/2.58
      = 4.0 + 1.27 + 2.5 + 0.43 + 1.16
      = 9.36
```

**Step 2: Calculate IDCG@5 (ideal ranking)**
```
IDCG@5 = 5/log₂(2) + 4/log₂(3) + 3/log₂(4) + 2/log₂(5) + 1/log₂(6)
       = 5/1.0 + 4/1.58 + 3/2.0 + 2/2.32 + 1/2.58
       = 5.0 + 2.53 + 1.5 + 0.86 + 0.39
       = 10.28
```

**Step 3: Calculate NDCG@5**
```
NDCG@5 = DCG@5 / IDCG@5 = 9.36 / 10.28 = 0.91
```

**Result:** NDCG@5 = 0.91 (Excellent ranking!)

#### NDCG Interpretation:
- **Range**: 0.0 to 1.0
- **1.0**: Perfect ranking (impossible to improve)
- **0.9+**: Outstanding ranking
- **0.8-0.9**: Excellent ranking  
- **0.7-0.8**: Very good ranking
- **0.6-0.7**: Good ranking
- **0.5-0.6**: Fair ranking
- **<0.5**: Poor ranking (needs improvement)

#### Why NDCG Dominates Ranking Evaluation:

1. **Position Awareness**: Heavily weights top positions where users focus
2. **Graded Relevance**: Handles multi-level relevance (not just binary)
3. **Normalization**: Enables fair comparison across different queries
4. **Industry Standard**: Used by Google, Amazon, Netflix, Spotify
5. **Intuitive**: Higher scores = better user experience

### Other Ranking Metrics

#### MAP (Mean Average Precision)
- **Use Case**: Binary relevance (relevant/not relevant)
- **Focus**: Precision at each relevant item position
- **Limitation**: Doesn't handle graded relevance well

#### MRR (Mean Reciprocal Rank)
- **Formula**: `MRR = 1/rank_of_first_relevant_item`
- **Use Case**: When only finding the first good result matters
- **Example**: Search queries where users want one answer

#### Precision@k / Recall@k
- **Precision@k**: What fraction of top-k results are relevant?
- **Recall@k**: What fraction of all relevant items are in top-k?
- **Limitation**: Treats all positions equally (no discounting)

#### Hit Rate@k
- **Definition**: Percentage of queries with at least one relevant item in top-k
- **Use Case**: Measuring coverage of recommendation systems

### Choosing the Right Metric

| Metric | Best For | Pros | Cons |
|--------|----------|------|------|
| **NDCG@k** | Most scenarios | Position-aware, graded relevance | More complex to interpret |
| **MAP** | Binary relevance | Well-established, intuitive | No position discounting |
| **MRR** | Single answer needed | Simple, fast to compute | Only cares about first result |
| **Precision@k** | Top-k performance | Simple to understand | No relevance grading |

### Practical NDCG Guidelines

#### Common NDCG@k Values:
- **NDCG@1**: How good is your top recommendation?
- **NDCG@5**: How good are your top 5 recommendations?
- **NDCG@10**: How good is your first page of results?

#### Industry Benchmarks:
- **E-commerce**: NDCG@10 > 0.7 considered good
- **Search Engines**: NDCG@10 > 0.8 expected
- **Movie/Music**: NDCG@5 > 0.6 acceptable for cold start users

#### NDCG Optimization Tips:
1. **Focus on top positions**: Getting top 3 items right has huge impact
2. **Quality over quantity**: Better to have fewer high-quality recommendations
3. **Handle ties**: When relevance scores are equal, secondary signals matter
4. **Monitor by user segment**: New users vs. existing users have different NDCG patterns

## 4. Making Inference with New Data

### Inference Process

#### Step 1: Prepare New Data
```python
# New user profile
new_user_data = {
    'user_id': 'user_new',
    'avg_rating': 3.5,
    'rating_std': 1.2,
    'num_purchases': 15
}

# Candidate items to rank
candidate_items = [
    {'product_id': 0, 'product_name': 'Device A Pro', 'category': 'Device_A', 'price': 299, ...},
    {'product_id': 1, 'product_name': 'Device B Wireless', 'category': 'Device_B', 'price': 59, ...},
    {'product_id': 2, 'product_name': 'Service Ultimate', 'category': 'Service', 'price': 15, ...}
]
```

#### Step 2: Feature Engineering
```python
# Create features for each user-product pair
inference_features = []
for product in candidate_items:
    features = create_features(new_user_data, product)
    inference_features.append(features)

X_inference = np.array(inference_features)
```

#### Step 3: Generate Predictions
```python
# Get relevance scores for all candidates
scores = model.predict(X_inference)
# scores = [-0.23, 0.45, -0.12]  # Example output
```

#### Step 4: Rank and Recommend
```python
# Sort products by predicted score (descending)
product_score_pairs = list(zip(candidate_items, scores))
ranked_products = sorted(product_score_pairs, key=lambda x: x[1], reverse=True)

# Extract top-k recommendations
top_k_recommendations = [product for product, score in ranked_products[:5]]
```

### Inference Best Practices

#### Feature Consistency:
- Use identical feature engineering pipeline as training
- Handle missing features gracefully
- Scale features if training data was scaled

#### Cold Start Problem:
- **New Users**: Use demographic features, default preferences
- **New Items**: Use item features, category averages
- **Hybrid Approaches**: Combine content-based with collaborative filtering

#### Real-time Considerations:
- Cache user features for fast lookup
- Pre-compute item features
- Use approximate methods for large candidate sets
- Implement feature stores for production systems

### Production Pipeline Example:
```python
def recommend_for_user(user_id, candidate_products, model, top_k=10):
    """Generate recommendations for a user."""
    
    # 1. Get user features
    user_features = get_user_features(user_id)
    
    # 2. Create user-product feature combinations
    X_candidates = []
    for product in candidate_products:
        features = combine_user_product_features(user_features, product)
        X_candidates.append(features)
    
    # 3. Predict scores
    scores = model.predict(np.array(X_candidates))
    
    # 4. Rank and return top-k
    product_scores = list(zip(candidate_products, scores))
    ranked_products = sorted(product_scores, key=lambda x: x[1], reverse=True)
    
    return [product for product, score in ranked_products[:top_k]]
```

## 5. Model Interpretation

### Understanding Predictions:
- **Absolute Values**: Don't matter (can be negative)
- **Relative Order**: What matters for ranking
- **Score Differences**: Larger gaps indicate stronger preferences

### Feature Importance:
```python
# Get feature importance from trained model
importance = model.feature_importance(importance_type='gain')
feature_names = ['price', 'category', 'user_rating', ...]

for name, imp in zip(feature_names, importance):
    print(f"{name}: {imp}")
```

### Troubleshooting Common Issues:

#### Poor NDCG Scores:
- Check feature quality and relevance
- Verify group sizes (need multiple items per query)
- Ensure sufficient training data per user/query
- Review label quality and distribution

#### Overfitting:
- Reduce model complexity (max_depth, num_leaves)
- Increase regularization (lambda_l1, lambda_l2)  
- Use early stopping with validation set
- Ensure sufficient training data diversity

#### Cold Start Performance:
- Include content-based features
- Use demographic and categorical features
- Implement fallback ranking strategies
- Consider hybrid recommendation approaches

## Summary

Learn-to-rank with LightGBM requires:
1. **Structured data** with query groups, features, and relevance labels
2. **Pairwise training** that learns relative item preferences  
3. **NDCG evaluation** to measure ranking quality
4. **Consistent feature engineering** for reliable inference
5. **Proper handling** of new users and items in production

The key insight is that ranking models optimize **relative ordering** rather than absolute prediction accuracy, making them ideal for recommendation systems where the goal is to present the best items first.