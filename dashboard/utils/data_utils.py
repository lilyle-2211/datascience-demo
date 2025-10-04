"""Data utilities for the dashboard."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def create_synthetic_ltv_data(n_samples=10000):
    """
    Create synthetic customer LTV dataset with realistic features.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Synthetic LTV dataset
    """
    np.random.seed(42)
    
    # Customer demographics
    age = np.random.normal(40, 12, n_samples).clip(18, 70)
    income = np.random.normal(60000, 20000, n_samples).clip(25000, 150000)
    
    # Account characteristics
    tenure_months = np.random.exponential(18, n_samples).clip(1, 60)
    monthly_spend = np.random.normal(150, 80, n_samples).clip(20, 500)
    
    # Service usage
    transactions_per_month = np.random.poisson(8, n_samples).clip(1, 30)
    support_tickets = np.random.poisson(2, n_samples).clip(0, 15)
    
    # Categorical features
    customer_segment = np.random.choice(['Basic', 'Premium', 'Enterprise'], 
                                      n_samples, p=[0.6, 0.3, 0.1])
    acquisition_channel = np.random.choice(['Online', 'Referral', 'Direct Sales', 'Partner'],
                                         n_samples, p=[0.4, 0.25, 0.2, 0.15])
    geography = np.random.choice(['Urban', 'Suburban', 'Rural'], 
                               n_samples, p=[0.5, 0.35, 0.15])
    
    # Binary features
    is_premium_customer = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    has_mobile_app = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    uses_autopay = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    has_referred_others = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Create LTV based on realistic factors
    base_ltv = 1000  # Base LTV
    
    # Income effect (higher income = higher LTV)
    ltv = base_ltv + (income - 60000) * 0.02
    
    # Tenure effect (longer tenure = higher LTV)
    ltv += tenure_months * 50
    
    # Monthly spend effect (higher spend = higher LTV)
    ltv += monthly_spend * 8
    
    # Age effect (middle-aged customers typically have higher LTV)
    age_factor = 1 + 0.02 * (age - 40) - 0.0005 * (age - 40)**2
    ltv *= age_factor
    
    # Segment effect
    segment_multiplier = {'Basic': 1.0, 'Premium': 1.5, 'Enterprise': 2.5}
    ltv *= [segment_multiplier[seg] for seg in customer_segment]
    
    # Channel effect
    channel_multiplier = {'Online': 1.0, 'Referral': 1.3, 'Direct Sales': 1.1, 'Partner': 0.9}
    ltv *= [channel_multiplier[ch] for ch in acquisition_channel]
    
    # Geography effect
    geo_multiplier = {'Urban': 1.1, 'Suburban': 1.0, 'Rural': 0.9}
    ltv *= [geo_multiplier[geo] for geo in geography]
    
    # Binary feature effects
    ltv *= (1 + is_premium_customer * 0.3)
    ltv *= (1 + has_mobile_app * 0.15)
    ltv *= (1 + uses_autopay * 0.1)
    ltv *= (1 + has_referred_others * 0.25)
    
    # Transaction frequency effect
    ltv += transactions_per_month * 100
    
    # Support tickets (negative effect)
    ltv -= support_tickets * 200
    
    # Add some noise
    ltv += np.random.normal(0, ltv * 0.1, n_samples)
    
    # Ensure positive LTV
    ltv = np.maximum(ltv, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': [f'C{i:06d}' for i in range(n_samples)],
        'age': age.round().astype(int),
        'income': income.round().astype(int),
        'tenure_months': tenure_months.round().astype(int),
        'monthly_spend': monthly_spend.round(2),
        'transactions_per_month': transactions_per_month,
        'support_tickets': support_tickets,
        'customer_segment': customer_segment,
        'acquisition_channel': acquisition_channel,
        'geography': geography,
        'is_premium_customer': is_premium_customer,
        'has_mobile_app': has_mobile_app,
        'uses_autopay': uses_autopay,
        'has_referred_others': has_referred_others,
        'ltv': ltv.round(2)
    })
    
    return data

def preprocess_ltv_data(df):
    """
    Preprocess the LTV data for model training.
    
    Args:
        df (pd.DataFrame): Raw LTV dataset
        
    Returns:
        tuple: Preprocessed features (X) and target (y)
    """
    data = df.copy()
    
    # Feature engineering
    data['spend_per_transaction'] = data['monthly_spend'] / data['transactions_per_month']
    data['income_per_age'] = data['income'] / data['age']
    data['tenure_income_ratio'] = data['tenure_months'] / (data['income'] / 1000)
    
    # Encode categorical variables
    le_segment = LabelEncoder()
    le_channel = LabelEncoder() 
    le_geography = LabelEncoder()
    
    data['segment_encoded'] = le_segment.fit_transform(data['customer_segment'])
    data['channel_encoded'] = le_channel.fit_transform(data['acquisition_channel'])
    data['geography_encoded'] = le_geography.fit_transform(data['geography'])
    
    # Select features for modeling
    feature_columns = [
        'age', 'income', 'tenure_months', 'monthly_spend',
        'transactions_per_month', 'support_tickets',
        'spend_per_transaction', 'income_per_age', 'tenure_income_ratio',
        'segment_encoded', 'channel_encoded', 'geography_encoded',
        'is_premium_customer', 'has_mobile_app', 'uses_autopay', 'has_referred_others'
    ]
    
    X = data[feature_columns]
    y = data['ltv']
    
    return X, y

def prepare_ml_data(sample_size=10000, test_size=0.2, random_state=42):
    """
    Prepare complete ML dataset with train/test split.
    
    Args:
        sample_size (int): Number of samples to generate
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (df, X_train, X_test, y_train, y_test)
    """
    df = create_synthetic_ltv_data(sample_size)
    X, y = preprocess_ltv_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return df, X_train, X_test, y_train, y_test