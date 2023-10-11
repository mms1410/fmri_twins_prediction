"""Compare labels for data trained on and new data."""
import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.data.twins_graphs_dataset import TwinsConnectomeDataset

twins_dataset = TwinsConnectomeDataset(factor_non_twins=1)
twins_dataset_less_twins = TwinsConnectomeDataset(factor_non_twins=5)


baseline_labels = twins_dataset.df_metadata["are_twins"]
current_labels = twins_dataset_less_twins.df_metadata["are_twins"]

data_baseline = pd.DataFrame(
    {
        "labels": baseline_labels
        # , "probs": baseline_probs}
    }
)
data_current = pd.DataFrame(
    {
        "labels": current_labels
        # , "probs": current_probs}
    }
)

# Create Evidently Report
drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
drift_report.run(reference_data=data_baseline, current_data=data_current)

# To display the HTML report in Jupyter notebook or Colab, call the object:
drift_report

# To export HTML as a separate file:
drift_report.save_html("evidently_drift_report.html")
