"""Regression model UI components."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

def display_metrics_comparison(train_metrics, test_metrics, model_name):
    """Display metrics comparison table."""
    metrics_df = pd.DataFrame({
        'Metric': ['R²', 'MSE', 'RMSE', 'MAE', 'MAPE'],
        'Training': [
            f"{train_metrics['r2']:.4f}",
            f"${train_metrics['mse']:,.0f}",
            f"${train_metrics['rmse']:,.2f}",
            f"${train_metrics['mae']:,.2f}",
            f"{train_metrics['mape']:.2%}"
        ],
        'Test': [
            f"{test_metrics['r2']:.4f}",
            f"${test_metrics['mse']:,.0f}",
            f"${test_metrics['rmse']:,.2f}",
            f"${test_metrics['mae']:,.2f}",
            f"{test_metrics['mape']:.2%}"
        ],
        'Difference': [
            f"{abs(train_metrics['r2'] - test_metrics['r2']):.4f}",
            f"${abs(train_metrics['mse'] - test_metrics['mse']):,.0f}",
            f"${abs(train_metrics['rmse'] - test_metrics['rmse']):,.2f}",
            f"${abs(train_metrics['mae'] - test_metrics['mae']):,.2f}",
            f"{abs(train_metrics['mape'] - test_metrics['mape']):.2%}"
        ]
    })
    
    st.subheader(f"{model_name} Performance Metrics")
    st.dataframe(metrics_df, use_container_width=True)
    
    # Overfitting analysis
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    rmse_diff_pct = ((test_metrics['rmse'] - train_metrics['rmse']) / train_metrics['rmse']) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Drop (Train→Test)", f"{r2_diff:.4f}")
    with col2:
        st.metric("RMSE Increase", f"{rmse_diff_pct:.1f}%")

def create_regression_plots(y_test, predictions, model_name):
    """Create regression evaluation plots."""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Actual vs Predicted', 'Residuals vs Predicted', 
                       'Residuals Distribution', 'Q-Q Plot'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=y_test, y=predictions, mode='markers', 
                  name='Predictions', opacity=0.6),
        row=1, col=1
    )
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name='Perfect Prediction', 
                  line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Residuals vs Predicted
    residuals = y_test - predictions
    fig.add_trace(
        go.Scatter(x=predictions, y=residuals, mode='markers', 
                  name='Residuals', opacity=0.6),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    # Residuals Distribution
    fig.add_trace(
        go.Histogram(x=residuals, name='Residuals Distribution', nbinsx=50),
        row=2, col=1
    )
    
    # Q-Q Plot (simplified)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                  mode='markers', name='Q-Q Plot', opacity=0.6),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'{model_name} - Regression Evaluation Plots',
        height=800,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Actual LTV ($)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted LTV ($)", row=1, col=1)
    fig.update_xaxes(title_text="Predicted LTV ($)", row=1, col=2)
    fig.update_yaxes(title_text="Residuals ($)", row=1, col=2)
    fig.update_xaxes(title_text="Residuals ($)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
    
    return fig

def display_business_context(test_metrics, y_test):
    """Display business context metrics."""
    st.subheader("Business Context")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Actual LTV", f"${y_test.mean():,.2f}")
    with col2:
        st.metric("Median Actual LTV", f"${y_test.median():,.2f}")
    with col3:
        st.metric("RMSE as % of Mean", f"{(test_metrics['rmse'] / y_test.mean()) * 100:.1f}%")
    with col4:
        st.metric("Prediction Range", 
                 f"${test_metrics['predictions'].min():,.0f} - ${test_metrics['predictions'].max():,.0f}")

def display_feature_importance_linear(model, feature_names, title="Feature Importance"):
    """Display feature importance for linear regression."""
    st.subheader(title)
    coefficients = model.coef_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    fig_importance = px.bar(
        importance_df.head(10), 
        x='Abs_Coefficient', 
        y='Feature', 
        orientation='h',
        title='Top 10 Most Important Features (by Absolute Coefficient)',
        color='Coefficient',
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_importance, use_container_width=True)

def display_feature_importance_xgboost(model, feature_names, title="Feature Importance"):
    """Display feature importance for XGBoost."""
    st.subheader(title)
    importance_scores = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        importance_df.head(10), 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Top 10 Most Important Features (XGBoost)',
        color='Importance',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_importance, use_container_width=True)