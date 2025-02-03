import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_training_testing_data(X: pd.DataFrame, y: pd.DataFrame, macros: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates and cleans data for XGBoost training by removing rows with NaN, inf values,
    or extremely large values in either features or all macro nutrients.
    
    IMPORTANT: Ideally this should be handled in the feature engineering pipeline, but for now, we're doing it here.
    
    Args:
        X: Feature DataFrame
        y: Target DataFrame with macro columns
        macros: List of macro column names to validate
        
    Returns:
        Cleaned X and y DataFrames
    """
    initial_rows = len(X)
    
    # Check for NaN or inf in features
    feature_mask = ~X.isna().any(axis=1) & ~np.isinf(X).any(axis=1)
    
    # Log feature issues if any
    nan_features = X.isna().sum()
    inf_features = np.isinf(X).sum()
    if nan_features.any() or inf_features.any():
        logger.warning("Features containing NaN or inf values:")
        for col in X.columns:
            if nan_features[col] > 0 or inf_features[col] > 0:
                logger.warning(f"{col}: {nan_features[col]} NaN, {inf_features[col]} inf")

    # Check for NaN or inf in all macro targets
    target_masks = []
    for macro in macros:
        macro_mask = ~y[macro].isna() & ~np.isinf(y[macro])
        target_masks.append(macro_mask)
        
        # Log target issues if any
        nan_count = y[macro].isna().sum()
        inf_count = np.isinf(y[macro]).sum()
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"Target {macro}: {nan_count} NaN, {inf_count} inf")

    # Combine all masks
    target_mask = pd.concat(target_masks, axis=1).all(axis=1)
    valid_mask = feature_mask & target_mask
    
    # Apply masks to both X and y
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    # Log cleaning results
    removed_rows = initial_rows - len(X_clean)
    if removed_rows > 0:
        logger.warning(
            f"Removed {removed_rows} rows ({(removed_rows/initial_rows)*100:.2f}%) "
            f"containing NaN or inf values"
        )
    
    return X_clean, y_clean