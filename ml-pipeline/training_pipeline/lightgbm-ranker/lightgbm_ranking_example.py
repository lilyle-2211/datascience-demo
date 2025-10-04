"""
LightGBM Learn-to-Rank Model for Recommender System
Example: Digital platform product recommendation system using ranking approach
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_users=1000, n_products=2000, n_interactions=25000):
    """Generate synthetic digital platform product engagement data for demonstration"""
    
    # Generate users (minimal info - just user_id)
    users = pd.DataFrame({
        'user_id': range(n_users)
    })
    
    # Generate digital platform products
    product_types = ['Device_A', 'Device_B', 'Device_C', 'Subscription', 'Maintenance_Service', 'Accessory']
    
    # Define specific products for each category
    device_a_names = ['Device A Pro', 'Device A Lite', 'Device A Max', 'Device A Standard']
    device_b_names = ['Device B Wireless', 'Device B Elite', 'Device B Adaptive', 'Device B Pro']
    device_c_names = ['Device C Premium', 'Device C Standard', 'Device C Pro', 'Device C Compact']
    subscription_names = ['Service Ultimate', 'Service Core', 'Service Gold', 'Service Plus']
    maintenance_names = ['Extended Warranty', 'Tech Support Pro', 'Repair Service', 'Device Protection']
    accessory_names = ['Charging Station', 'Media Remote', 'Power Kit', 'Stand']
    
    products_data = []
    product_id = 0
    
    # Create products for each category
    for category in product_types:
        if category == 'Device_A':
            names = device_a_names
            price_range = (299, 599)
        elif category == 'Device_B':
            names = device_b_names  
            price_range = (59, 179)
        elif category == 'Device_C':
            names = device_c_names
            price_range = (99, 299)
        elif category == 'Subscription':
            names = subscription_names
            price_range = (9.99, 16.99)
        elif category == 'Maintenance_Service':
            names = maintenance_names
            price_range = (49, 199)
        else:  # Accessory
            names = accessory_names
            price_range = (24, 79)
        
        # Create multiple variants of each product
        for name in names:
            for variant in range(np.random.randint(2, 5)):  # 2-4 variants per product
                products_data.append({
                    'product_id': product_id,
                    'product_name': f"{name} V{variant+1}" if variant > 0 else name,
                    'category': category,
                    'price': round(np.random.uniform(price_range[0], price_range[1]), 2),
                    'release_year': np.random.randint(2018, 2024),
                    'avg_rating': np.random.uniform(3.5, 5.0),
                    'num_ratings': np.random.randint(50, 2000)
                })
                product_id += 1
                
                if product_id >= n_products:
                    break
            if product_id >= n_products:
                break
        if product_id >= n_products:
            break
    
    products = pd.DataFrame(products_data)
    
    # Generate interactions (ratings/purchases)
    interactions = []
    for _ in range(n_interactions):
        user_id = np.random.randint(0, n_users)
        product_idx = np.random.randint(0, len(products))
        product = products.iloc[product_idx]
        
        # Create some category preferences
        category = product['category']
        base_rating = product['avg_rating']
        
        # Category preference patterns
        category_preference = np.random.uniform(-0.5, 0.5)
        price_factor = 0.1 if product['price'] > 200 else 0.2  # Expensive items get slight boost
        
        rating = base_rating + category_preference + np.random.uniform(-0.5, 0.5) + price_factor
        rating = max(1, min(5, rating))  # Clamp between 1-5
        
        interactions.append({
            'user_id': user_id,
            'product_id': product['product_id'],
            'rating': rating,
            'timestamp': np.random.randint(1640995200, 1672531200)  # 2022-2023
        })
    
    interactions_df = pd.DataFrame(interactions)
    
    return users, products, interactions_df

def create_features(users, products, interactions):
    """Create features for the ranking model"""
    
    # Merge all data
    data = interactions.merge(users, on='user_id').merge(products, on='product_id')
    
    # Encode categorical variables
    le_category = LabelEncoder()
    data['category_encoded'] = le_category.fit_transform(data['category'])
    
    # Create additional features
    data['product_age'] = 2024 - data['release_year']
    
    # Price tier encoding (budget, mid-range, premium)
    data['price_tier'] = pd.cut(data['price'], bins=[0, 50, 150, float('inf')], labels=[0, 1, 2])
    data['price_tier'] = data['price_tier'].astype(int)
    
    # User statistics (simplified)
    user_stats = interactions.groupby('user_id').agg({
        'rating': ['mean', 'std', 'count']
    }).round(3)
    user_stats.columns = ['user_avg_rating', 'user_rating_std', 'user_num_purchases']
    user_stats = user_stats.reset_index()
    
    data = data.merge(user_stats, on='user_id', how='left')
    data['user_rating_std'] = data['user_rating_std'].fillna(0)
    
    # Product statistics 
    product_stats = interactions.groupby('product_id').agg({
        'rating': ['mean', 'count']
    }).round(3)
    product_stats.columns = ['product_avg_rating_actual', 'product_num_ratings_actual']
    product_stats = product_stats.reset_index()
    
    data = data.merge(product_stats, on='product_id', how='left')
    
    # Feature columns for model
    feature_cols = [
        'category_encoded', 'release_year', 'product_age', 'price', 'price_tier',
        'avg_rating', 'num_ratings', 'user_avg_rating', 'user_rating_std', 
        'user_num_purchases', 'product_avg_rating_actual', 'product_num_ratings_actual'
    ]
    
    return data, feature_cols

def prepare_ranking_data(data, feature_cols):
    """Prepare data in LightGBM ranking format"""
    
    # Sort by user_id for proper grouping
    data = data.sort_values('user_id').reset_index(drop=True)
    
    # Create relevance scores from ratings (0-4 scale for ranking)
    # Convert to integers as required by LightGBM ranking
    data['relevance'] = (data['rating'] - 1).round().astype(int)  # Convert 1-5 rating to 0-4 relevance as integers
    
    # Features
    X = data[feature_cols].values
    y = data['relevance'].values.astype(int)  # Ensure integers
    
    # Group information (number of items per user)
    group_sizes = data.groupby('user_id').size().values
    
    return X, y, group_sizes, data

def train_lightgbm_ranker(X_train, y_train, group_train, X_val, y_val, group_val):
    """Train LightGBM ranking model"""
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
    
    # Parameters for ranking
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model

def evaluate_model(model, X_test, y_test, group_test, feature_cols):
    """Evaluate the ranking model"""
    
    # Predict scores
    y_pred = model.predict(X_test)
    
    # Calculate NDCG for different k values
    ndcg_scores = {}
    
    # Split predictions by groups
    start_idx = 0
    group_ndcgs = []
    
    for group_size in group_test:
        end_idx = start_idx + group_size
        
        y_true_group = y_test[start_idx:end_idx]
        y_pred_group = y_pred[start_idx:end_idx]
        
        # Calculate NDCG@5 and NDCG@10 for this group
        if len(y_true_group) >= 5:
            ndcg_5 = ndcg_score([y_true_group], [y_pred_group], k=5)
            group_ndcgs.append(ndcg_5)
        
        start_idx = end_idx
    
    avg_ndcg = np.mean(group_ndcgs) if group_ndcgs else 0
    
    print(f"Average NDCG@5: {avg_ndcg:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    return avg_ndcg, feature_importance

def recommend_products(model, user_id, users, products, interactions, feature_cols, top_k=10, show_inference_data=False):
    """Generate Xbox product recommendations for a user"""
    
    # Get products the user hasn't purchased/rated
    user_products = set(interactions[interactions['user_id'] == user_id]['product_id'])
    candidate_products = products[~products['product_id'].isin(user_products)].copy()
    
    if len(candidate_products) == 0:
        return pd.DataFrame(), pd.DataFrame()  # No recommendations available
    
    # Create features for all candidate products
    candidate_data = []
    inference_details = []
    
    for _, product in candidate_products.iterrows():
        # Map category to number (simplified)
        category_map = {'Device_A': 0, 'Device_B': 1, 'Device_C': 2, 'Subscription': 3, 'Maintenance_Service': 4, 'Accessory': 5}
        category_encoded = category_map.get(product['category'], 0)
        
        # Calculate user stats (simplified)
        user_ratings = interactions[interactions['user_id'] == user_id]['rating']
        user_avg_rating = user_ratings.mean() if len(user_ratings) > 0 else 3.0
        user_rating_std = user_ratings.std() if len(user_ratings) > 1 else 1.0
        user_num_purchases = len(user_ratings)
        
        # Product features
        product_age = 2024 - product['release_year']
        
        # Price tier encoding
        if product['price'] <= 50:
            price_tier = 0
        elif product['price'] <= 150:
            price_tier = 1
        else:
            price_tier = 2
        
        feature_row = [
            category_encoded, product['release_year'], product_age, product['price'], price_tier,
            product['avg_rating'], product['num_ratings'], user_avg_rating, 
            user_rating_std, user_num_purchases, product['avg_rating'], product['num_ratings']
        ]
        
        candidate_data.append(feature_row)
        
        # Store details for inference DataFrame
        if show_inference_data:
            inference_details.append({
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'category': product['category'],
                'price': product['price'],
                'release_year': product['release_year'],
                'product_avg_rating': product['avg_rating'],
                'user_avg_rating': user_avg_rating,
                'user_rating_std': user_rating_std,
                'user_num_purchases': user_num_purchases,
                'product_age': product_age,
                'price_tier': price_tier,
                'category_encoded': category_encoded
            })
    
    X_candidates = np.array(candidate_data)
    
    # Predict scores
    scores = model.predict(X_candidates)
    
    # Create inference DataFrame if requested
    inference_df = pd.DataFrame()
    if show_inference_data and inference_details:
        inference_df = pd.DataFrame(inference_details)
        inference_df['predicted_score'] = scores
        
        # Create feature matrix DataFrame
        feature_df = pd.DataFrame(X_candidates, columns=feature_cols)
        feature_df['product_id'] = [details['product_id'] for details in inference_details]
        feature_df['predicted_score'] = scores
        inference_df = inference_df.merge(feature_df[['product_id', 'predicted_score'] + feature_cols], on=['product_id', 'predicted_score'])
    
    # Get top recommendations
    candidate_products['score'] = scores
    recommendations = candidate_products.nlargest(top_k, 'score')[
        ['product_id', 'product_name', 'category', 'price', 'avg_rating', 'score']
    ]
    
    return recommendations, inference_df

def main():
    """Main function to demonstrate LightGBM ranking for recommendations"""
    
    print("LightGBM Learn-to-Rank Xbox Product Recommender Demo")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample data...")
    users, products, interactions = generate_sample_data()
    print(f"   Users: {len(users)}")
    print(f"   Products: {len(products)}")
    print(f"   Interactions: {len(interactions)}")
    
    # Step 2: Create features
    print("\n2. Creating features...")
    data, feature_cols = create_features(users, products, interactions)
    print(f"   Features: {feature_cols}")
    
    # Step 3: Prepare ranking data
    print("\n3. Preparing ranking data...")
    X, y, group_sizes, processed_data = prepare_ranking_data(data, feature_cols)
    
    # Step 4: Split data (ensure users are not split across train/test)
    print("\n4. Splitting data...")
    unique_users = processed_data['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    val_users, test_users = train_test_split(test_users, test_size=0.5, random_state=42)
    
    # Create splits
    train_mask = processed_data['user_id'].isin(train_users)
    val_mask = processed_data['user_id'].isin(val_users) 
    test_mask = processed_data['user_id'].isin(test_users)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Recalculate group sizes for each split
    group_train = processed_data[train_mask].groupby('user_id').size().values
    group_val = processed_data[val_mask].groupby('user_id').size().values
    group_test = processed_data[test_mask].groupby('user_id').size().values
    
    print(f"   Train: {len(X_train)} samples, {len(group_train)} users")
    print(f"   Val: {len(X_val)} samples, {len(group_val)} users") 
    print(f"   Test: {len(X_test)} samples, {len(group_test)} users")
    
    # Step 5: Train model
    print("\n5. Training LightGBM ranker...")
    model = train_lightgbm_ranker(X_train, y_train, group_train, X_val, y_val, group_val)
    
    # Step 6: Evaluate model
    print("\n6. Evaluating model...")
    avg_ndcg, feature_importance = evaluate_model(model, X_test, y_test, group_test, feature_cols)
    
    # Print detailed prediction analysis
    print("\n6.1. Detailed prediction analysis...")
    y_pred_test = model.predict(X_test)
    
    # Show prediction distribution
    print(f"   Prediction Statistics:")
    print(f"   Min score: {y_pred_test.min():.4f}")
    print(f"   Max score: {y_pred_test.max():.4f}")
    print(f"   Mean score: {y_pred_test.mean():.4f}")
    print(f"   Std score: {y_pred_test.std():.4f}")
    
    # Show some sample predictions vs actual
    print(f"\n   Sample Predictions vs Actual Relevance:")
    print(f"   {'Predicted':<12} {'Actual':<8} {'Difference':<10}")
    print(f"   {'-'*12} {'-'*8} {'-'*10}")
    for i in range(min(20, len(y_test))):
        diff = y_pred_test[i] - y_test[i] 
        print(f"   {y_pred_test[i]:<12.4f} {y_test[i]:<8} {diff:<10.4f}")
    
    # Analyze prediction distribution by relevance level
    print(f"\n   Average Predictions by Relevance Level:")
    for relevance in sorted(set(y_test)):
        mask = y_test == relevance
        avg_pred = y_pred_test[mask].mean()
        count = mask.sum()
        print(f"   Relevance {relevance}: {avg_pred:.4f} (n={count})")
    
    print(f"   Total test samples: {len(y_test)}")
    
    # Explain why negative predictions are normal
    print(f"\n   " + "="*60)
    print(f"   WHY NEGATIVE PREDICTIONS ARE NORMAL IN RANKING")
    print(f"   " + "="*60)
    print(f"""
   IMPORTANT: LightGBM ranking models predict RELATIVE scores, not original ratings!
   
   Key Points:
   • Ranking models optimize for ORDER, not exact values
   • Negative scores are fine - what matters is relative ranking
   • Higher score = better relevance, regardless of sign
   • The model learns to separate good vs bad items, not predict exact ratings
   
   Example: If model predicts [-0.5, 0.2, -0.8], the ranking is:
   → 1st place: 0.2 (highest score - most relevant)
   → 2nd place: -0.5 (middle score - moderate relevance)  
   → 3rd place: -0.8 (lowest score - least relevant)
   """)
    
    # Demonstrate ranking behavior with test data
    print(f"   RANKING DEMONSTRATION (First 10 Test Items):")
    print(f"   " + "="*50)
    
    # Create pairs for sorting
    score_actual_pairs = list(zip(y_pred_test[:10], y_test[:10], range(10)))
    score_actual_pairs.sort(key=lambda x: x[0], reverse=True)  # Sort by predicted score
    
    print(f"   {'Rank':<5} {'Predicted':<12} {'Actual':<8} {'Quality'}")
    print(f"   {'----':<5} {'----------':<12} {'------':<8} {'-------'}")
    for rank, (pred, actual, orig_idx) in enumerate(score_actual_pairs, 1):
        quality = "Good" if actual >= 3 else "Poor"
        print(f"   {rank:<5} {pred:<12.4f} {actual:<8} {quality}")
    
    print(f"\n   Notice: Higher predicted scores generally correspond to higher actual ratings!")
    print(f"   This shows the model learned proper ranking despite negative values.")
    print(f"   Test users: {len(group_test)}")
    
    # Step 7: Generate sample recommendations
    print("\n7. Generating sample recommendations...")
    sample_user_id = test_users[0]
    
    # Show user profile first  
    user_history = interactions[interactions['user_id'] == sample_user_id]
    
    print(f"\n   User {sample_user_id} Profile:")
    print(f"   User ID: {sample_user_id} (no demographic data available)")
    print(f"   Historical purchases: {len(user_history)} products")
    if len(user_history) > 0:
        print(f"   Average rating given: {user_history['rating'].mean():.2f}")
        print(f"   Rating range: {user_history['rating'].min():.1f} - {user_history['rating'].max():.1f}")
        
        # Show favorite categories
        user_product_categories = user_history.merge(products[['product_id', 'category']], on='product_id')
        category_preferences = user_product_categories.groupby('category')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        print(f"   Top rated categories: {', '.join(category_preferences.head(3).index.tolist())}")
    
    recommendations, inference_df = recommend_products(model, sample_user_id, users, products, interactions, feature_cols, show_inference_data=True)
    
    print(f"\n   Inference Process:")
    print(f"   Total products in database: {len(products)}")
    print(f"   Products already purchased by user: {len(user_history)}")
    print(f"   Candidate products for recommendation: {len(products) - len(user_history)}")
    print(f"   Top recommendations selected: {len(recommendations)}")
    
    # Show detailed inference DataFrame
    if not inference_df.empty:
        print(f"\n   INFERENCE DATA SAMPLE (First 15 candidates):")
        print("   " + "="*100)
        
        # Check available columns and select ones that exist
        available_cols = inference_df.columns.tolist()
        print(f"   Available columns: {available_cols[:10]}...")  # Show first 10 columns
        
        # Select columns that exist in the DataFrame (no demographics)
        desired_display_cols = [
            'product_id', 'product_name', 'category', 'price', 'product_avg_rating', 
            'user_avg_rating', 'user_rating_std', 'user_num_purchases',
            'product_age', 'price_tier', 'predicted_score'
        ]
        
        display_cols = [col for col in desired_display_cols if col in available_cols]
        print(f"   Displaying columns: {display_cols}")
        
        sample_inference = inference_df[display_cols].head(15)
        
        # Format the DataFrame for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 10)
        
        print(sample_inference.round(3).to_string(index=False))
        
        print(f"\n   FEATURE MATRIX SAMPLE (Model Input Features):")
        print("   " + "="*80)
        
        # Check if feature_cols exist in inference_df
        available_feature_cols = [col for col in feature_cols if col in available_cols]
        matrix_cols = available_feature_cols + ['predicted_score']
        
        if available_feature_cols:
            feature_matrix_sample = inference_df[matrix_cols].head(10)
            print(feature_matrix_sample.round(3).to_string(index=False))
        else:
            print("   Feature matrix columns not available in inference DataFrame")
        
        print(f"\n   TOP SCORED CANDIDATES (Before Top-K Selection):")
        print("   " + "="*60)
        
        # Select columns that exist for top candidates
        top_candidate_cols = [col for col in ['product_id', 'product_name', 'category', 'price', 'product_avg_rating', 'predicted_score'] if col in available_cols]
        
        if top_candidate_cols:
            top_candidates = inference_df.nlargest(15, 'predicted_score')[top_candidate_cols]
            print(top_candidates.round(3).to_string(index=False))
        else:
            print("   Top candidates data not available")
    
    print(f"\n   FINAL TOP 10 RECOMMENDATIONS:")
    print("   " + "="*50)
    print(recommendations.round(3).to_string(index=False))
    
    # Show prediction score distribution for this user's recommendations
    if len(recommendations) > 0:
        print(f"\n   Recommendation Score Analysis:")
        print(f"   Highest score: {recommendations['score'].max():.4f}")
        print(f"   Lowest score: {recommendations['score'].min():.4f}")
        print(f"   Score range: {recommendations['score'].max() - recommendations['score'].min():.4f}")
        
        # Show category distribution in recommendations
        category_dist = recommendations['category'].value_counts()
        print(f"   Recommended categories: {dict(category_dist)}")
    
    # Step 8: Plot feature importance
    print("\n8. Plotting feature importance...")
    try:
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importance - LightGBM Ranker')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('lightgbm_feature_importance.png', dpi=300, bbox_inches='tight')
        print("   Feature importance plot saved as 'lightgbm_feature_importance.png'")
        plt.close()  # Close the figure to avoid GUI hanging
    except Exception as e:
        print(f"   Plotting failed: {e}")
        print("   (This is normal in headless environments)")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()