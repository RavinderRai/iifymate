from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def get_xgb_macro_model(X_train, X_test, y_train, y_test, macro, args):
    """
    Trains an XGBoost regressor model for predicting a specific macronutrient (carbs, fat, or protein) 
    using the recipe data from the Edamam API. Returns the trained model along with evaluation metrics.

    Parameters:
    X_train : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.
    X_test : array-like or sparse matrix, shape (n_samples, n_features)
        Test data.
    y_train : DataFrame, shape (n_samples, n_targets)
        Target values for training data.
    y_test : DataFrame, shape (n_samples, n_targets)
        Target values for test data.
    macro : str
        Name of the target macronutrient variable (column) in y_train and y_test.
    args : dict
        Dictionary containing arguments to be passed to the XGBRegressor constructor.

    Returns:
    xgb_model : XGBRegressor object
        Trained XGBoost regressor model.
    r2 : float
        R-squared score on the test data.
    mse : float
        Mean squared error on the test data.
    """
    xgb = XGBRegressor(**args)
    xgb.fit(X_train, y_train[macro])
    y_pred = xgb.predict(X_test)
    r2 = r2_score(y_test[macro], y_pred)
    mse = mean_squared_error(y_test[macro], y_pred)

    return xgb, r2, mse



if __name__ == "__main__":
    fat_args = {}
    carbs_args = {}
    protein_args = {}

    fat_xgb, fat_r2, fat_mse = get_xgb_macro_model(X_train, X_test, y_train, y_test, 'fat', fat_args)
    carbs_xgb, carbs_r2, carbs_mse = get_xgb_macro_model(X_train, X_test, y_train, y_test, 'carbs', carbs_args)
    protein_xgb, protein_r2, protein_mse = get_xgb_macro_model(X_train, X_test, y_train, y_test, 'protein', protein_args)