"""
Parametric Distribution Models - Functional Approach
===================================================

This module demonstrates parametric modeling techniques using:
1. Gaussian Mixture Models (GMM) from sklearn
2. Poisson Distribution from scipy.stats

All functions are standalone without classes for simplicity.

Author: Data Science Demo
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for saving plots  # noqa: E402

from typing import Dict, List, Tuple  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from data_generators import (  # noqa: E402
    generate_gaussian_mixture_data,
    generate_poisson_data,
)
from scipy.stats import chisquare, poisson  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def fit_gaussian_mixture(
    data: np.ndarray, n_components: int = 2, random_state: int = 42
) -> Tuple[GaussianMixture, Dict]:
    """
    Fit Gaussian Mixture Model to data.

    Args:
        data: Input data array
        n_components: Number of Gaussian components
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (fitted_model, parameters_dict)
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data)

    parameters = {
        "means": gmm.means_.flatten(),
        "covariances": gmm.covariances_.flatten(),
        "weights": gmm.weights_,
        "n_components": n_components,
        "log_likelihood": gmm.score(data),
    }

    return gmm, parameters


def predict_gmm_components(gmm_model: GaussianMixture, data: np.ndarray) -> np.ndarray:
    """
    Predict component membership for data points.

    Args:
        gmm_model: Fitted GMM model
        data: Input data array

    Returns:
        Component predictions
    """
    return gmm_model.predict(data)


def plot_gaussian_mixture(
    data: np.ndarray,
    gmm_model: GaussianMixture,
    parameters: Dict,
    save_path: str = None,
) -> None:
    """
    Plot the data distribution with fitted GMM components.

    Args:
        data: Original data
        gmm_model: Fitted GMM model
        parameters: Model parameters dictionary
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Plot histogram of original data
    plt.hist(
        data.flatten(),
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        label="Data Histogram",
    )

    # Plot individual components
    x_range = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)

    colors = ["red", "green", "orange", "purple"]

    for i in range(parameters["n_components"]):
        # Create individual Gaussian component
        component_density = (
            parameters["weights"][i]
            * (1 / np.sqrt(2 * np.pi * parameters["covariances"][i]))
            * np.exp(
                -0.5
                * (x_range.flatten() - parameters["means"][i]) ** 2
                / parameters["covariances"][i]
            )
        )

        plt.plot(
            x_range,
            component_density,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f'Component {i+1} (μ={parameters["means"][i]:.2f})',
        )

    # Plot overall mixture
    mixture_density = np.exp(gmm_model.score_samples(x_range))
    plt.plot(x_range, mixture_density, "black", linewidth=3, label="GMM Mixture")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f'Gaussian Mixture Model ({parameters["n_components"]} components)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close figure to free memory
    else:
        plt.show()


def fit_poisson_distribution(data: np.ndarray) -> float:
    """
    Fit Poisson distribution to data using MLE.

    Args:
        data: Count data array

    Returns:
        Estimated lambda parameter
    """
    return np.mean(data)


def calculate_poisson_statistics(data: np.ndarray) -> Dict:
    """
    Calculate Poisson distribution statistics.

    Args:
        data: Count data array

    Returns:
        Dictionary of statistics
    """
    lambda_est = fit_poisson_distribution(data)

    return {
        "lambda_estimated": lambda_est,
        "mean": np.mean(data),
        "variance": np.var(data),
        "theoretical_variance": lambda_est,  # For Poisson, var = mean
        "variance_ratio": np.var(data) / lambda_est,  # Should be ~1 for Poisson
        "sample_size": len(data),
        "min_value": data.min(),
        "max_value": data.max(),
    }


def poisson_goodness_of_fit(data: np.ndarray, lambda_param: float = None) -> Dict:
    """
    Perform goodness-of-fit test for Poisson distribution.

    Args:
        data: Count data array
        lambda_param: Lambda parameter (if None, estimated from data)

    Returns:
        Test statistics and p-value
    """
    if lambda_param is None:
        lambda_param = fit_poisson_distribution(data)

    # Observed frequencies
    unique_vals, observed_counts = np.unique(data, return_counts=True)

    # Expected frequencies
    expected_counts = []
    for val in unique_vals:
        expected_prob = poisson.pmf(val, mu=lambda_param)
        expected_counts.append(expected_prob * len(data))

    expected_counts = np.array(expected_counts)

    # Normalize to ensure sum consistency (fix numerical precision issues)
    expected_counts = expected_counts * (observed_counts.sum() / expected_counts.sum())

    # Chi-square test
    chi2_stat, p_value = chisquare(observed_counts, expected_counts)

    return {
        "chi2_statistic": chi2_stat,
        "p_value": p_value,
        "degrees_of_freedom": len(unique_vals) - 1 - 1,  # -1 for estimated parameter
        "interpretation": "Good fit" if p_value > 0.05 else "Poor fit",
        "lambda_used": lambda_param,
    }


def plot_poisson_distribution(
    data: np.ndarray, lambda_param: float = None, save_path: str = None
) -> None:
    """
    Plot Poisson distribution with fitted parameters.

    Args:
        data: Count data array
        lambda_param: Lambda parameter (if None, estimated from data)
        save_path: Optional path to save the plot
    """
    if lambda_param is None:
        lambda_param = fit_poisson_distribution(data)

    plt.figure(figsize=(12, 8))

    # Plot histogram
    counts, bins, _ = plt.hist(
        data,
        bins=range(int(data.min()), int(data.max()) + 2),
        density=True,
        alpha=0.7,
        color="lightcoral",
        label="Observed Data",
    )

    # Plot theoretical Poisson PMF
    x_vals = range(int(data.min()), int(data.max()) + 1)
    theoretical_pmf = [poisson.pmf(x, mu=lambda_param) for x in x_vals]

    plt.plot(
        x_vals,
        theoretical_pmf,
        "bo-",
        linewidth=2,
        markersize=8,
        label=f"Poisson PMF (λ={lambda_param:.2f})",
    )

    plt.xlabel("Count")
    plt.ylabel("Probability")
    plt.title("Poisson Distribution Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close figure to free memory
    else:
        plt.show()


def analyze_poisson_data(data: np.ndarray, plot: bool = True) -> Dict:
    """
    Complete Poisson analysis pipeline.

    Args:
        data: Count data array
        plot: Whether to create plots

    Returns:
        Complete analysis results
    """
    # Calculate statistics
    stats = calculate_poisson_statistics(data)

    # Goodness of fit test
    gof_results = poisson_goodness_of_fit(data, stats["lambda_estimated"])

    # Plot if requested
    if plot:
        plot_poisson_distribution(data, stats["lambda_estimated"])

    # Combine results
    results = {
        "statistics": stats,
        "goodness_of_fit": gof_results,
        "summary": {
            "is_good_fit": gof_results["p_value"] > 0.05,
            "variance_check": "PASS"
            if 0.8 <= stats["variance_ratio"] <= 1.5
            else "FAIL",
            "recommended_use": "Suitable for Poisson modeling"
            if gof_results["p_value"] > 0.05
            else "Consider alternative distributions",
        },
    }

    return results


def analyze_gaussian_mixture(
    data: np.ndarray, n_components: int = 2, plot: bool = True
) -> Dict:
    """
    Complete Gaussian Mixture analysis pipeline.

    Args:
        data: Input data array
        n_components: Number of components to fit
        plot: Whether to create plots

    Returns:
        Complete analysis results
    """
    # Fit model
    gmm_model, parameters = fit_gaussian_mixture(data, n_components)

    # Predict components
    component_labels = predict_gmm_components(gmm_model, data)

    # Plot if requested
    if plot:
        plot_gaussian_mixture(data, gmm_model, parameters)

    # Calculate component statistics
    component_stats = {}
    for i in range(n_components):
        mask = component_labels == i
        component_data = data[mask]
        component_stats[f"component_{i+1}"] = {
            "size": len(component_data),
            "proportion": len(component_data) / len(data),
            "mean": np.mean(component_data),
            "std": np.std(component_data),
        }

    results = {
        "model": gmm_model,
        "parameters": parameters,
        "component_labels": component_labels,
        "component_statistics": component_stats,
        "model_selection": {
            "aic": gmm_model.aic(data),
            "bic": gmm_model.bic(data),
            "log_likelihood": parameters["log_likelihood"],
        },
    }

    return results


def demonstrate_parametric_models(save_plots=True):
    """
    Demonstrate both Gaussian Mixture Models and Poisson distribution analysis.

    Args:
        save_plots: Whether to save example plots for documentation
    """
    print("=" * 60)
    print("PARAMETRIC DISTRIBUTION MODELS DEMONSTRATION")
    print("=" * 60)

    # 1. Gaussian Mixture Model Example
    print("\n1. GAUSSIAN MIXTURE MODEL ANALYSIS")
    print("-" * 40)

    # Generate and analyze GMM data
    gmm_data = generate_gaussian_mixture_data(n_samples=1000, random_state=42)
    gmm_results = analyze_gaussian_mixture(gmm_data, n_components=2, plot=False)

    # Save GMM plot for README
    if save_plots:
        plot_gaussian_mixture(
            gmm_data,
            gmm_results["model"],
            gmm_results["parameters"],
            save_path="gmm_example.png",
        )
        print("  📊 GMM plot saved as 'gmm_example.png'")

    params = gmm_results["parameters"]
    print(f"Fitted GMM Parameters:")
    for i in range(len(params["means"])):
        print(
            f"  Component {i+1}: μ={params['means'][i]:.2f}, "
            f"σ²={params['covariances'][i]:.2f}, "
            f"weight={params['weights'][i]:.2f}"
        )

    print(f"Model Selection Metrics:")
    print(f"  AIC: {gmm_results['model_selection']['aic']:.2f}")
    print(f"  BIC: {gmm_results['model_selection']['bic']:.2f}")

    # 2. Poisson Distribution Example
    print("\n2. POISSON DISTRIBUTION ANALYSIS")
    print("-" * 40)

    # Generate and analyze Poisson data
    poisson_data = generate_poisson_data(lam=4.2, size=1000, random_state=42)
    poisson_results = analyze_poisson_data(poisson_data, plot=False)

    # Save Poisson plot for README
    if save_plots:
        plot_poisson_distribution(
            poisson_data,
            poisson_results["statistics"]["lambda_estimated"],
            save_path="poisson_example.png",
        )
        print("  📊 Poisson plot saved as 'poisson_example.png'")

    print("Poisson Statistics:")
    for key, value in poisson_results["statistics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nGoodness of Fit Test:")
    gof = poisson_results["goodness_of_fit"]
    print(f"  Chi-square statistic: {gof['chi2_statistic']:.3f}")
    print(f"  P-value: {gof['p_value']:.3f}")
    print(f"  Interpretation: {gof['interpretation']}")

    print(f"\nSummary:")
    summary = poisson_results["summary"]
    print(f"  Good fit: {summary['is_good_fit']}")
    print(f"  Variance check: {summary['variance_check']}")
    print(f"  Recommendation: {summary['recommended_use']}")

    return gmm_results, poisson_results


if __name__ == "__main__":
    # Run the demonstration and save example plots
    gmm_results, poisson_results = demonstrate_parametric_models(save_plots=True)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nExample plots generated for documentation:")
    print("  - gmm_example.png")
    print("  - poisson_example.png")
    print("\nResults are available as 'gmm_results' and 'poisson_results'")
    print("Use individual functions for custom analysis:")
    print("  - fit_gaussian_mixture()")
    print("  - analyze_poisson_data()")
    print("  - generate_gaussian_mixture_data()")
    print("  - generate_poisson_data()")
