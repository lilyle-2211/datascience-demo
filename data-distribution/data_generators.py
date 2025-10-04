"""
Example Data Generators
======================

This module contains functions to generate synthetic data for testing
parametric distribution models.
"""

import numpy as np
from scipy.stats import poisson


def generate_gaussian_mixture_data(
    n_samples: int = 1000,
    means: list = [5.0, 15.0],
    stds: list = [1.5, 2.0],
    weights: list = [0.5, 0.5],
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate synthetic multimodal data from Gaussian mixture.

    Args:
        n_samples: Total number of samples to generate
        means: List of component means
        stds: List of component standard deviations
        weights: List of component weights (should sum to 1)
        random_state: Random seed for reproducibility

    Returns:
        Generated data array
    """
    np.random.seed(random_state)

    if len(means) != len(stds) or len(means) != len(weights):
        raise ValueError("means, stds, and weights must have same length")

    if not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must sum to 1.0")

    # Generate samples for each component
    all_samples = []

    for mean, std, weight in zip(means, stds, weights):
        n_component_samples = int(n_samples * weight)
        component_samples = np.random.normal(
            loc=mean, scale=std, size=n_component_samples
        )
        all_samples.extend(component_samples)

    # Handle any rounding issues
    while len(all_samples) < n_samples:
        # Add samples from first component
        extra_sample = np.random.normal(loc=means[0], scale=stds[0], size=1)
        all_samples.extend(extra_sample)

    # Convert to array and shuffle
    data = np.array(all_samples[:n_samples])
    np.random.shuffle(data)

    return data.reshape(-1, 1)


def generate_poisson_data(
    lam: float = 3.5, size: int = 1000, random_state: int = 42
) -> np.ndarray:
    """
    Generate synthetic Poisson-distributed data.

    Args:
        lam: Lambda parameter (rate)
        size: Number of samples
        random_state: Random seed for reproducibility

    Returns:
        Generated count data
    """
    np.random.seed(random_state)
    return poisson.rvs(mu=lam, size=size)


def generate_customer_arrivals(
    lam_hourly: float = 8.5, hours: int = 24 * 7, random_state: int = 42
) -> dict:
    """
    Generate synthetic customer arrival data (Poisson process).

    Args:
        lam_hourly: Average customers per hour
        hours: Number of hours to simulate
        random_state: Random seed

    Returns:
        Dictionary with hourly and daily aggregated data
    """
    np.random.seed(random_state)

    # Generate hourly arrivals
    hourly_arrivals = poisson.rvs(mu=lam_hourly, size=hours)

    # Aggregate to daily totals (assuming 24-hour days)
    days = hours // 24
    daily_totals = hourly_arrivals[: days * 24].reshape(days, 24).sum(axis=1)

    return {
        "hourly_arrivals": hourly_arrivals,
        "daily_totals": daily_totals,
        "total_customers": hourly_arrivals.sum(),
        "avg_per_hour": hourly_arrivals.mean(),
        "avg_per_day": daily_totals.mean(),
    }


def generate_bimodal_sales_data(
    n_samples: int = 1000,
    low_season_mean: float = 25.0,
    high_season_mean: float = 75.0,
    low_season_std: float = 8.0,
    high_season_std: float = 15.0,
    high_season_weight: float = 0.3,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate bimodal sales data representing seasonal patterns.

    Args:
        n_samples: Number of sales records
        low_season_mean: Average sales during low season
        high_season_mean: Average sales during high season
        low_season_std: Standard deviation for low season
        high_season_std: Standard deviation for high season
        high_season_weight: Proportion of high season (0-1)
        random_state: Random seed

    Returns:
        Sales data array
    """
    return generate_gaussian_mixture_data(
        n_samples=n_samples,
        means=[low_season_mean, high_season_mean],
        stds=[low_season_std, high_season_std],
        weights=[1 - high_season_weight, high_season_weight],
        random_state=random_state,
    )


def generate_defect_counts(
    lam_per_batch: float = 2.3, n_batches: int = 500, random_state: int = 42
) -> dict:
    """
    Generate manufacturing defect count data.

    Args:
        lam_per_batch: Average defects per batch
        n_batches: Number of production batches
        random_state: Random seed

    Returns:
        Dictionary with defect statistics
    """
    np.random.seed(random_state)

    defect_counts = poisson.rvs(mu=lam_per_batch, size=n_batches)

    return {
        "defect_counts": defect_counts,
        "total_defects": defect_counts.sum(),
        "defect_rate": defect_counts.mean(),
        "zero_defect_batches": (defect_counts == 0).sum(),
        "max_defects_in_batch": defect_counts.max(),
        "batches_analyzed": n_batches,
    }


if __name__ == "__main__":
    # Test the data generators
    print("Testing Data Generators")
    print("=" * 40)

    # Test Gaussian mixture
    gmm_data = generate_gaussian_mixture_data(n_samples=100)
    print(f"GMM data shape: {gmm_data.shape}")
    print(f"GMM data range: {gmm_data.min():.2f} to {gmm_data.max():.2f}")

    # Test Poisson
    poisson_data = generate_poisson_data(lam=4.0, size=100)
    print(f"Poisson data range: {poisson_data.min()} to {poisson_data.max()}")

    # Test customer arrivals
    customer_data = generate_customer_arrivals(hours=24)
    print(f"Customer arrivals per day: {customer_data['daily_totals']}")

    print("\nAll generators working correctly!")
