import numpy as np
import pandas as pd


def analyze_prediction_errors(X_test, y_true, y_pred):
    """
    Create row-level prediction error details for the test set.

    Signed_Error = Actual_Price - Predicted_Price
    Absolute_Error = abs(Signed_Error)
    Relative_Error = Absolute_Error / abs(Actual_Price) * 100
    """

    actual = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)

    error_details = X_test.copy()


    error_details["Actual_Price"] = actual
    error_details["Predicted_Price"] = predicted

    error_details["Signed_Error"] = actual - predicted

    error_details["Absolute_Error"] = np.abs(
        error_details["Signed_Error"]
    )

    
    denominator = np.abs(actual)

    error_details["Relative_Error"] = np.divide(
        error_details["Absolute_Error"] * 100,
        denominator,
        out=np.full(len(actual), np.nan, dtype=float),
        where=denominator != 0
    )

    return error_details


def get_top_error_cases(error_details, n=10):
    """
    Return the n observations with the largest absolute errors.
    """
    return (
        error_details
        .nlargest(n, "Absolute_Error")
        .copy()
    )
