import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_data_quality_metrics(df, df_ground_truth=None):
    """
    Compute key data quality metrics for a given DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
        df_ground_truth: optional reference DataFrame for accuracy calculation
    
    Returns:
        dict: Dictionary containing the metric values
    
    Raises:
        ValueError: If required columns are missing or data types are invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    metrics = {}
    
    # Completeness
    metrics["completeness"] = df.notnull().mean().mean()
    
    # Accuracy (only if ground truth is provided)
    if df_ground_truth is not None:
        if df.shape != df_ground_truth.shape:
            raise ValueError("DataFrame and ground truth must have same shape")
        metrics["accuracy"] = (df == df_ground_truth).mean().mean()
    
    # Timeliness (if timestamp column exists)
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            raise ValueError("timestamp column must be datetime type")
        metrics["timeliness"] = (
            df["timestamp"] >= pd.Timestamp.now() - pd.Timedelta(days=1)
        ).mean()
    
    # Consistency (only for string columns)
    string_cols = df.select_dtypes(include="object").columns
    if len(string_cols) > 0:
        consistency_scores = []
        for col in string_cols:
            if df[col].notna().any():  # Only check non-empty columns
                values = df[col].fillna('')
                # Check if values follow a consistent pattern (start with same character)
                first_chars = values.str[0]
                mode_char = first_chars.mode()[0] if not first_chars.empty else ''
                matches = values.str.startswith(mode_char).fillna(False)
                consistency_scores.append(matches.mean())
        metrics["consistency"] = np.mean(consistency_scores) if consistency_scores else 0.0
    
    # Uniqueness
    metrics["uniqueness"] = df.drop_duplicates().shape[0] / df.shape[0]
    
    return metrics

def create_sample_data():
    """
    Create sample data with intentional quality issues for testing.
    
    Returns:
        tuple: (DataFrame with quality issues, Ground truth DataFrame)
    """
    np.random.seed(42)
    current_time = pd.Timestamp.now()
    
    # Create initial data
    data = {
        'id': range(1000),
        'name': ['John Smith', 'Jane Doe', 'Bob Wilson', 'Alice Brown'] * 250,
        'age': np.random.randint(18, 80, 1000),
        'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com'] * 250,
        'timestamp': [current_time - timedelta(hours=np.random.randint(0, 48)) for _ in range(1000)]
    }
    
    # Create initial DataFrame
    df = pd.DataFrame(data)
    
    # Create ground truth before modifications
    df_ground_truth = df.copy()
    
    # Add missing values (affects completeness)
    missing_age_idx = np.random.choice(df.index, 50, replace=False)
    missing_email_idx = np.random.choice(df.index, 30, replace=False)
    df.loc[missing_age_idx, 'age'] = np.nan
    df.loc[missing_email_idx, 'email'] = np.nan
    
    # Add duplicates (affects uniqueness)
    duplicate_indices = np.random.choice(df.index, 100, replace=False)
    df = pd.concat([df, df.loc[duplicate_indices]], ignore_index=True)
    df_ground_truth = pd.concat([df_ground_truth, df_ground_truth.loc[duplicate_indices]], ignore_index=True)
    
    # Introduce some inaccuracies
    inaccurate_idx = np.random.choice(df.index, 100, replace=False)
    df.loc[inaccurate_idx, 'age'] = np.random.randint(18, 80, size=len(inaccurate_idx))
    
    return df, df_ground_truth

def test_data_quality_metrics():
    """
    Test the data quality metrics calculation with sample data.
    
    Returns:
        dict: Calculated metrics
    """
    print("Creating sample data...")
    df, df_ground_truth = create_sample_data()
    
    print("\nCalculating metrics...")
    metrics = calculate_data_quality_metrics(df, df_ground_truth)
    
    print("\nData Quality Metrics:")
    print("--------------------")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2%}")
    
    # Additional dataset information
    print("\nDataset Information:")
    print("-------------------")
    print(f"Total records: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    
    return metrics

if __name__ == "__main__":
    test_data_quality_metrics()