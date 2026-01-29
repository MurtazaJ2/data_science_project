import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import subprocess
from pathlib import Path

from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import save_json


# ===============================
# DagsHub MLflow credentials
# ===============================
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/MurtazaJ2/data_science_project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "MurtazaJ2"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8182909a1298d62fcaa19c5371d4fab7f6fd69e6"


# ===============================
# DVC version helper
# ===============================
def get_dvc_version():
    try:
        # Get current Git commit (this is the real data version)
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()

        return git_commit
    except Exception:
        return "unknown"



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):

        # Load test data & model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # Explicit experiment (important for DagsHub UI)
        mlflow.set_experiment("Wine Quality using ElasticNet")

        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        with mlflow.start_run():

            predictions = model.predict(test_x)
            # ===============================
            # Save & log predictions
            # ===============================
            pred_df = test_x.copy()
            pred_df["actual_quality"] = test_y.values
            pred_df["predicted_quality"] = predictions
            
            predictions_path = Path("artifacts/model_evaluation/predictions.csv")
            pred_df.to_csv(predictions_path, index=False)
            
            mlflow.log_artifact(str(predictions_path))
            
            rmse, mae, r2 = self.eval_metrics(test_y, predictions)

            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(
                path=Path(self.config.metric_file_name),
                data=scores
            )

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log params used for training
            mlflow.log_params(self.config.all_params)

            # 🔗 DVC dataset lineage
            mlflow.log_param("data_version", get_dvc_version())
            mlflow.log_artifact("artifacts/data_ingestion/winequality-red.csv.dvc")


            # Register model (DagsHub supports registry)
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="WineQualityElasticNet"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
