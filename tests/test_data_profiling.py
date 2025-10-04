"""
Test suite for DataProfiling class.

This module contains comprehensive tests for data profiling functionality
including null percentage analysis, zero value detection, statistical summaries,
distinct counts, distribution analysis, and outlier detection.
"""

import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

# Add the data-profiling directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "data-profiling"))

from data_profiling import DataProfiling

def test_simple_import():
    """Test that we can import the data profiling module."""
    from data_profiling import DataProfiling
    assert DataProfiling is not None


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    spark = (
        SparkSession.builder.appName("DataProfilingTest")
        .master("local[1]")  # Use only 1 core for faster startup
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Disable Arrow for faster startup
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def sample_df(spark):
    """Create a sample DataFrame for testing."""
    data = [
        (1, 30, 50000, "user_a", None),
        (2, 35, 60000, "user_b", "active"),
        (3, None, 70000, "user_c", "inactive"),
        (4, 0, 0, "user_d", "active"),  # Changed empty string to "user_d"
        (5, 25, 55000, "user_e", "active")
    ]
    columns = ["id", "num_rounds", "spend", "username", "status"]
    return spark.createDataFrame(data, columns)


@pytest.fixture(scope="module")  
def numeric_df(spark):
    """Create a DataFrame with numeric data for statistical testing."""
    data = [(1, 10.5), (2, 15.2), (3, 8.7), (4, 12.1), (5, 100.0)]  # 100.0 is an outlier
    columns = ["id", "value"]
    return spark.createDataFrame(data, columns)


def test_data_profiling_initialization(spark, sample_df):
    """Test DataProfiling class initialization."""
    # Test normal initialization
    profiler = DataProfiling(spark, sample_df)
    assert profiler._spark == spark
    assert profiler._df.count() == sample_df.count()
    
    # Test with excluded columns
    profiler_excluded = DataProfiling(spark, sample_df, excluded_col=["id"])
    assert "id" not in profiler_excluded._cols
    
    # Test with multiple excluded columns
    profiler_multi_excluded = DataProfiling(spark, sample_df, excluded_col=["id", "username"])
    assert "id" not in profiler_multi_excluded._cols
    assert "username" not in profiler_multi_excluded._cols


def test_data_profiling_initialization_errors(spark):
    """Test DataProfiling initialization with invalid inputs."""
    # Test with None DataFrame
    with pytest.raises(ValueError, match="DataFrame cannot be None"):
        DataProfiling(spark, None)


def test_get_zero_percent(spark, sample_df):
    """Test zero percentage calculation for numeric columns."""
    profiler = DataProfiling(spark, sample_df)
    result = profiler.get_zero_percent()
    
    # Verify result structure
    assert result.columns == ["Column", "ZeroPercentage"]
    
    # Convert to list for easier testing
    result_list = result.collect()
    result_dict = {row["Column"]: row["ZeroPercentage"] for row in result_list}
    
    # Should only include numeric columns
    numeric_columns = ["id", "num_rounds", "spend"]
    assert set(result_dict.keys()).issubset(set(numeric_columns))
    
    # Check specific zero percentages
    assert result_dict["num_rounds"] == "20.00%"  # 1 zero out of 5
    assert result_dict["spend"] == "20.00%"  # 1 zero out of 5


def test_get_statistics(spark, numeric_df):
    """Test statistical summary calculation."""
    profiler = DataProfiling(spark, numeric_df)
    result = profiler.get_statistics()
    
    # Result should be a list of DataFrames
    assert isinstance(result, list)
    assert len(result) > 0
    
    # First DataFrame should contain summary statistics
    summary_df = result[0]
    assert "summary" in summary_df.columns
    
    # Should have additional DataFrames for skewness/kurtosis of each numeric column
    assert len(result) >= 2  # At least summary + skewness/kurtosis for numeric columns


def test_get_distinct_counts(spark, sample_df):
    """Test distinct count calculation."""
    profiler = DataProfiling(spark, sample_df)
    result = profiler.get_distinct_counts()
    
    # Verify result structure
    assert result.columns == ["Column", "DistinctCount"]
    
    # Convert to list for easier testing
    result_list = result.collect()
    result_dict = {row["Column"]: int(row["DistinctCount"]) for row in result_list}
    
    # Check that all columns are included
    expected_columns = ["id", "num_rounds", "spend", "username", "status"]
    assert set(result_dict.keys()) == set(expected_columns)
    
    # Verify some specific counts
    assert result_dict["id"] == 5  # All unique IDs
    assert result_dict["status"] == 3  # "active", "inactive", null


def test_get_distribution_counts(spark, sample_df):
    """Test distribution count calculation."""
    profiler = DataProfiling(spark, sample_df)
    result = profiler.get_distribution_counts()
    
    # Result should be a list of DataFrames
    assert isinstance(result, list)
    assert len(result) == len(sample_df.columns)  # One DataFrame per column
    
    # Each DataFrame should have count column
    for df in result:
        assert "count" in df.columns


def test_flag_outliers(spark, numeric_df):
    """Test outlier detection using IQR method."""
    profiler = DataProfiling(spark, numeric_df)
    result = profiler.flag_outliers(["value"], factor=1.5)
    
    # Result should be a list of DataFrames
    assert isinstance(result, list)
    assert len(result) == 1  # One DataFrame for the "value" column
    
    # Check that outliers are detected
    outlier_df = result[0]
    outliers = outlier_df.collect()
    
    # Should detect the outlier value (100.0)
    assert len(outliers) > 0


def test_flag_outliers_with_invalid_columns(spark, sample_df):
    """Test outlier detection with invalid column names."""
    profiler = DataProfiling(spark, sample_df)
    
    # Test with non-existent column
    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        profiler.flag_outliers(["non_existent_column"])
    
    # Test with empty column list
    with pytest.raises(ValueError, match="outlier_detect_cols cannot be empty"):
        profiler.flag_outliers([])


def test_flag_outliers_custom_factor(spark, numeric_df):
    """Test outlier detection with custom factor."""
    profiler = DataProfiling(spark, numeric_df)
    
    # Test with different factors
    result_factor_1 = profiler.flag_outliers(["value"], factor=1.0)
    result_factor_2 = profiler.flag_outliers(["value"], factor=2.0)
    
    # More restrictive factor (1.0) should detect more outliers than less restrictive (2.0)
    outliers_factor_1 = result_factor_1[0].count()
    outliers_factor_2 = result_factor_2[0].count()
    
    assert outliers_factor_1 >= outliers_factor_2


def test_empty_dataframe_handling(spark):
    """Test handling of empty DataFrames."""
    # Create empty DataFrame with explicit schema for PySpark 4.0+
    from pyspark.sql.types import StructType, StructField, IntegerType
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("value", IntegerType(), True)
    ])
    empty_df = spark.createDataFrame([], schema)
    
    profiler = DataProfiling(spark, empty_df)
    
    # All methods should handle empty DataFrames gracefully
    zero_result = profiler.get_zero_percent()
    assert zero_result.count() == 0
    
    distinct_result = profiler.get_distinct_counts()
    # For empty DataFrame, distinct_counts still returns column info (2 columns = 2 rows)
    assert distinct_result.count() == 2  # One row per column


def test_categorical_and_numeric_column_identification(spark):
    """Test proper identification of categorical and numeric columns."""
    # Create DataFrame with mixed column types
    data = [
        (1, 25.5, "category_a", "2023-01-01"),
        (2, 30.0, "category_b", "2023-01-02"),
        (3, 28.7, "category_a", "2023-01-03")
    ]
    columns = ["id", "score", "category", "date"]
    df = spark.createDataFrame(data, columns)
    
    profiler = DataProfiling(spark, df)
    
    # Check categorical columns identification
    assert "category" in profiler._categorical_cols
    assert "date" in profiler._categorical_cols
    
    # Check numeric columns identification  
    assert "id" in profiler._numeric_cols
    assert "score" in profiler._numeric_cols


def test_column_exclusion_functionality(spark, sample_df):
    """Test column exclusion functionality in various scenarios."""
    # Test single column exclusion (string)
    profiler_single = DataProfiling(spark, sample_df, excluded_col="id")
    assert "id" not in profiler_single._cols
    assert len(profiler_single._cols) == len(sample_df.columns) - 1
    
    # Test single column exclusion (list)
    profiler_list = DataProfiling(spark, sample_df, excluded_col=["id"])
    assert "id" not in profiler_list._cols
    
    # Test multiple column exclusion
    profiler_multi = DataProfiling(spark, sample_df, excluded_col=["id", "username"])
    assert "id" not in profiler_multi._cols
    assert "username" not in profiler_multi._cols
    assert len(profiler_multi._cols) == len(sample_df.columns) - 2


if __name__ == "__main__":
    pytest.main([__file__])
