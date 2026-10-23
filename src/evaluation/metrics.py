import numpy as np

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_true - y_pred

    mae_error = np.mean(np.abs(error))
    mse_error = np.mean(error **2)
    rmse_error = np.sqrt(mse_error)

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        "MAE": mae_error,
        "MSE": mse_error,
        "RMSE": rmse_error,
        "R2": r_squared
    }
