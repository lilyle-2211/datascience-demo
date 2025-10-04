"""
Data profiling utilities for PySpark DataFrames.

This module provides comprehensive data profiling capabilities including
null percentage analysis, zero value detection, statistical summaries,
distinct counts, distribution analysis, and outlier detection.
"""

import logging
from typing import List, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


class DataProfiling:
    """
    Data profiling class for comprehensive DataFrame analysis.
    
    This class provides methods for analyzing data quality, distributions,
    and statistical properties of PySpark DataFrames.
    """
    
    def __init__(
        self, 
        spark: SparkSession, 
        df: DataFrame, 
        excluded_col: Optional[Union[str, int, List[Union[str, int]]]] = None
    ) -> None:
        """
        Initialize DataProfiling with SparkSession and DataFrame.
        
        Args:
            spark: SparkSession object for DataFrame operations
            df: Input DataFrame to profile
            excluded_col: Column(s) to exclude from analysis
            
        Raises:
            ValueError: If DataFrame is empty or invalid
        """
        super().__init__()
        self._spark = spark
        self._df = df
        self._excluded_col = excluded_col

        # Validate input DataFrame
        if df is None:
            raise ValueError("DataFrame cannot be None")
        
        try:
            df_count = self._df.count()
            if df_count == 0:
                logging.warning("DataFrame is empty")
        except Exception as e:
            raise ValueError(f"Invalid DataFrame provided: {e}")

        # Handle column exclusion
        if self._excluded_col is not None:
            if isinstance(self._excluded_col, (str, int)):
                excluded_cols = [self._excluded_col]
            else:
                excluded_cols = self._excluded_col
            
            self._cols = [
                col for col in self._df.columns if col not in excluded_cols
            ]
            self._df = self._df.select(*self._cols)
        else:
            self._cols = self._df.columns

        # Identify categorical columns (string type)
        self._categorical_cols = [
            item[0] for item in self._df.dtypes if item[1].startswith("string")
        ]
        logging.info(f"Categorical columns identified: {self._categorical_cols}")

        # Identify numeric columns (bigint, double, float, etc.)
        self._numeric_cols = [
            item[0] for item in self._df.dtypes 
            if item[1].startswith(("bigint", "double", "float", "int", "decimal"))
        ]
        logging.info(f"Numeric columns identified: {self._numeric_cols}")

    def get_zero_percent(self) -> DataFrame:
        """
        Calculate zero value percentages for numeric columns.

        Returns:
            DataFrame: DataFrame with columns 'Column' and 'ZeroPercentage'
            
        Raises:
            RuntimeError: If calculation fails due to DataFrame issues
            
        Calculation:
            For each numeric column, calculates percentage of zero values
            relative to total row count.
        """
        schema = StructType([
            StructField("Column", StringType(), True),
            StructField("ZeroPercentage", StringType(), True),
        ])
        
        try:
            empty_rdd = self._spark.sparkContext.emptyRDD()
            result_df = self._spark.createDataFrame(empty_rdd, schema=schema)
            df_count = self._df.count()
            
            if df_count == 0:
                logging.warning("DataFrame is empty, returning empty result")
                return result_df

            for col_name in self._numeric_cols:
                try:
                    zero_count = (
                        self._df.select(F.col(col_name))
                        .filter(F.col(col_name) == 0.0)
                        .count()
                    )
                    
                    zero_percentage = (zero_count * 100.0 / df_count)
                    zero_row = self._spark.createDataFrame(
                        [[col_name, f"{zero_percentage:.2f}%"]], schema=schema
                    )
                    result_df = result_df.union(zero_row)
                    
                except Exception as e:
                    logging.error(f"Error calculating zero percentage for column {col_name}: {e}")
                    # Add NaN entry for failed calculation
                    zero_row = self._spark.createDataFrame(
                        [[col_name, "NaN%"]], schema=schema
                    )
                    result_df = result_df.union(zero_row)
            
            return result_df
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate zero percentages: {e}")

    def get_statistics(self) -> List[DataFrame]:
        """
        Calculate statistical summaries for numeric columns.

        Returns:
            List[DataFrame]: List of DataFrames containing skewness and kurtosis
            for each numeric column
            
        Raises:
            RuntimeError: If statistical calculation fails
            
        Calculation:
            - Standard summary statistics (count, mean, stddev, min, max)
            - Skewness (measure of asymmetry)
            - Kurtosis (measure of tail heaviness)
        """
        try:
            result_dataframes = []
            
            if not self._numeric_cols:
                logging.warning("No numeric columns found for statistical analysis")
                return result_dataframes
            
            # Display standard summary statistics
            try:
                summary_df = self._df.select(self._numeric_cols).summary()
                logging.info("Standard summary statistics calculated successfully")
                result_dataframes.append(summary_df)
            except Exception as e:
                logging.error(f"Error calculating summary statistics: {e}")

            # Calculate skewness and kurtosis for each numeric column
            for col_name in self._numeric_cols:
                try:
                    stats_df = self._df.select(
                        F.skewness(col_name).alias(f"skewness_{col_name}"),
                        F.kurtosis(col_name).alias(f"kurtosis_{col_name}"),
                    )
                    result_dataframes.append(stats_df)
                    
                except Exception as e:
                    logging.error(f"Error calculating skewness/kurtosis for column {col_name}: {e}")
                    continue

            return result_dataframes
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate statistics: {e}")

    def get_distinct_counts(self) -> DataFrame:
        """
        Calculate distinct value counts for all columns.

        Returns:
            DataFrame: DataFrame with columns 'Column' and 'DistinctCount'
            
        Raises:
            RuntimeError: If distinct count calculation fails
            
        Calculation:
            For each column, counts the number of unique values.
        """
        schema = StructType([
            StructField("Column", StringType(), True),
            StructField("DistinctCount", StringType(), True),
        ])

        try:
            empty_rdd = self._spark.sparkContext.emptyRDD()
            result_df = self._spark.createDataFrame(empty_rdd, schema=schema)

            for col_name in self._cols:
                try:
                    distinct_count = self._df.select(F.col(col_name)).distinct().count()
                    distinct_row = self._spark.createDataFrame(
                        [[col_name, str(distinct_count)]], schema=schema
                    )
                    result_df = result_df.union(distinct_row)
                    
                except Exception as e:
                    logging.error(f"Error calculating distinct count for column {col_name}: {e}")
                    # Add error entry for failed calculation
                    distinct_row = self._spark.createDataFrame(
                        [[col_name, "Error"]], schema=schema
                    )
                    result_df = result_df.union(distinct_row)
            
            return result_df
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate distinct counts: {e}")

    def get_distribution_counts(self) -> List[DataFrame]:
        """
        Calculate value distribution counts for all columns.

        Returns:
            List[DataFrame]: List of DataFrames with value counts for each column,
            sorted by count in descending order
            
        Raises:
            RuntimeError: If distribution calculation fails
            
        Calculation:
            For each column, groups by unique values and counts occurrences,
            then sorts by count in descending order.
        """
        try:
            result_dataframes = []
            
            for col_name in self._cols:
                try:
                    distribution_df = (
                        self._df.groupby(F.col(col_name))
                        .count()
                        .sort(F.col("count").desc())
                    )
                    result_dataframes.append(distribution_df)
                    
                except Exception as e:
                    logging.error(f"Error calculating distribution for column {col_name}: {e}")
                    continue
            
            return result_dataframes
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate distribution counts: {e}")

    def flag_outliers(
        self, 
        outlier_detect_cols: List[str], 
        factor: Optional[float] = None
    ) -> List[DataFrame]:
        """
        Detect outliers using Inter-Quartile Range (IQR) proximity rule.

        Args:
            outlier_detect_cols: List of column names to analyze for outliers
            factor: IQR multiplication factor for outlier detection (default: 1.5)

        Returns:
            List[DataFrame]: List of DataFrames containing outliers for each column
            
        Raises:
            ValueError: If invalid columns provided or calculation fails
            RuntimeError: If outlier detection process fails
            
        Calculation:
            Uses IQR proximity rule where outliers are values that fall below 
            Q1 - factor*IQR or above Q3 + factor*IQR, where Q1 and Q3 are 
            the 25th and 75th percentiles respectively.
        """
        if factor is None:
            factor = 1.5
            
        if not outlier_detect_cols:
            raise ValueError("outlier_detect_cols cannot be empty")
            
        # Validate that all columns exist in DataFrame
        missing_cols = [col for col in outlier_detect_cols if col not in self._df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        logging.info(
            f"Using Inter-Quartile Range (IQR) proximity rule to detect outliers. "
            f"This method is recommended for statistically skewed distributions."
        )
        logging.info(
            f"Data points that fall below Q1 - {factor} IQR or above Q3 + {factor} IQR "
            f"are considered outliers, where Q1 and Q3 are the 25th and 75th percentiles."
        )

        try:
            # Calculate quartile bounds for each column
            bounds = {}
            for col_name in outlier_detect_cols:
                if col_name in self._df.columns:
                    try:
                        quartiles = self._df.approxQuantile(col_name, [0.25, 0.75], 0)
                        bounds[col_name] = {"q1": quartiles[0], "q3": quartiles[1]}
                    except Exception as e:
                        logging.error(f"Error calculating quartiles for column {col_name}: {e}")
                        continue

            # Calculate outlier bounds
            for col_name in bounds:
                try:
                    iqr = bounds[col_name]["q3"] - bounds[col_name]["q1"]
                    bounds[col_name]["min"] = bounds[col_name]["q1"] - (iqr * factor)
                    bounds[col_name]["max"] = bounds[col_name]["q3"] + (iqr * factor)
                    
                    logging.info(
                        f"Column {col_name} - Lower bound: {bounds[col_name]['min']:.4f}; "
                        f"Upper bound: {bounds[col_name]['max']:.4f}; Factor: {factor}"
                    )
                except Exception as e:
                    logging.error(f"Error calculating bounds for column {col_name}: {e}")
                    continue

            # Create outlier detection columns
            outlier_cols = [
                F.when(
                    ~F.col(col_name).between(bounds[col_name]["min"], bounds[col_name]["max"]), 
                    F.col(col_name)
                ).alias(f"{col_name}_outlier")
                for col_name in bounds
            ]

            if not outlier_cols:
                logging.warning("No valid columns for outlier detection")
                return []

            # Extract outliers for each column
            outlier_df = self._df.select(*outlier_cols)
            result_dataframes = []
            
            for outlier_col in outlier_df.columns:
                try:
                    column_outliers = (
                        outlier_df.select(outlier_col)
                        .filter(F.col(outlier_col).isNotNull())
                    )
                    result_dataframes.append(column_outliers)
                except Exception as e:
                    logging.error(f"Error extracting outliers for {outlier_col}: {e}")
                    continue
            
            return result_dataframes
            
        except Exception as e:
            raise RuntimeError(f"Failed to detect outliers: {e}")
