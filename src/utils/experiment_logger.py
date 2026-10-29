from pathlib import Path
from datetime import datetime
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "report" / "model_results.csv"


def save_experiment(
    model,
    target,
    feature_configuration,
    hyperparameters,
    metrics,
):
    """Append one experiment's results to the CSV immediately."""

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "Model": model,
        "Target": target,
        "Feature_Configuration": feature_configuration,
        "Hyperparameters": json.dumps(hyperparameters),
        **metrics,
    }

    results_df = pd.DataFrame([record])

    results_df.to_csv(
        RESULTS_PATH,
        mode="a",
        header=not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0,
        index=False,
    )

    print(f"Experiment saved to: {RESULTS_PATH}")
