"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X = np.asarray(X, dtype=float)

    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0, col_mean)

    inds = np.where(np.isnan(X))
    X[inds] = col_mean[inds[1]]

    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    lower, upper = [], []
    for col in range(X.shape[1]):
        q1 = np.percentile(X[:, col], 25)
        q3 = np.percentile(X[:, col], 75)
        iqr = q3 - q1
        lower.append(q1 - k * iqr)
        upper.append(q3 + k * iqr)
    return lower, upper

# Step 3 - clip_columns
import numpy as np
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    # for col in range(X.shape[1]):
    #     X[:, col] = np.clip(X[:, col], lower[col], upper[col])
    # return X

    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    new_X = X.copy()
    new_X = np.append(new_X, col.reshape(-1, 1), axis=1)
    return new_X

# Step 6 - one_hot_encode
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(labels):
    # sorted_cat = np.unique(labels)
    # ohe = OneHotEncoder(categories=[sorted_cat], dtype=np.float64, sparse_output=False)
    # return ohe.fit_transform(labels.reshape(-1, 1))

    categories, inverse = np.unique(labels, return_inverse=True)
    one_hot = np.zeros((labels.size, categories.size))
    one_hot[np.arange(labels.size), inverse] = 1
    return one_hot

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    # bias = np.ones((X.shape[0], 1))
    # new_X = np.append(bias, X, axis=1)
    # return new_X

    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    np.random.seed(seed)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return indices

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test sets.
    N = len(indices)
    t_idx = int(N * train_ratio)
    v_idx = int(N * val_ratio)
    train_idx = indices[:t_idx]
    val_idx = indices[t_idx:t_idx + v_idx]
    test_idx = indices[t_idx + v_idx:]
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    x_sub = X[indices]
    y_sub = y[indices]
    return x_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A = X.T @ X
    b = X.T @ y
    theta = np.linalg.solve(A, b)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_true - y_pred))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_res = np.sum((y_true - y_pred) ** 2)
    mean_y = np.mean(y_true)
    SS_tot = np.sum((y_true - mean_y) ** 2)
    if SS_tot == 0.0:
        return 0.0
    r2 = 1 - (SS_res / SS_tot)
    return r2

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        return None
    if len(y_true) != len(y_pred):
        return None
    residuals = y_true - y_pred
    r_mean = np.mean(residuals)
    r_std = np.std(residuals, ddof=0)
    r_median = np.median(np.abs(residuals))
    return {
        "mean": r_mean,
        "std": r_std,
        "median_abs": r_median,
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X, iqr_k)
    X = clip_columns(X, lower, upper)
    return X

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    ratio_col = make_ratio_feature(numerator, denominator)
    X_num = append_column(X_num, ratio_col)
    if cat_labels is not None:
        cat_labels = one_hot_encode(cat_labels)
        X_num = np.append(X_num, cat_labels, axis=1)
    return X_num

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    N = X.shape[0]
    idx = make_shuffled_indices(N, seed)
    train_idx, val_idx, test_idx = partition_indices(idx, train_ratio, val_ratio)
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)
    return {
        "X_train": X_train, 
        "y_train": y_train, 
        "X_val": X_val, 
        "y_val": y_val, 
        "X_test": X_test, 
        "y_test": y_test,
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    mean, std = fit_standardizer(splits["X_train"])
    std_splits = {}
    for model in ["X_train", "X_val", "X_test"]:
        std_splits[model] = apply_standardizer(splits[model], mean, std)
        std_splits[model] = add_bias_column(std_splits[model])
    std_splits["y_train"] = splits["y_train"]
    std_splits["y_val"] = splits["y_val"]
    std_splits["y_test"] = splits["y_test"]
    return std_splits, mean, std

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

