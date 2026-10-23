
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "report" / "model_results.csv"


def save_experiment(experiment):
    """Append one experiment record to the results CSV."""

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame([experiment])

    df.to_csv(
        RESULTS_PATH,
        mode="a",
        header=(
            not RESULTS_PATH.exists()
            or RESULTS_PATH.stat().st_size == 0
        ),
        index=False
    )

    print(f"Experiment saved: {experiment.get('Experiment', 'Unnamed')}")
