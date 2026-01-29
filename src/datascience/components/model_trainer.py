import pandas as pd
import os
from src.datascience import logger
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
import joblib
import mlflow
import mlflow.sklearn

from src.datascience.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        param_grid = {
            "alpha": [0.1, 0.5, 1.0],
            "l1_ratio": [0.1, 0.5, 0.9]
        }

        best_r2 = -1
        best_model = None

        mlflow.set_experiment("Wine Quality using ElasticNet")

        for params in ParameterGrid(param_grid):
            with mlflow.start_run(run_name="elasticnet_tuning"):
                logger.info(f"Training with params: {params}")

                model = ElasticNet(**params, random_state=42)
                model.fit(train_x, train_y)

                preds = model.predict(test_x)
                r2 = r2_score(test_y, preds)

                mlflow.log_params(params)
                mlflow.log_metric("r2", r2)

                mlflow.sklearn.log_model(model, "model")

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model

        joblib.dump(
            best_model,
            os.path.join(self.config.root_dir, self.config.model_name)
        )

        logger.info(f"Best model saved with R2: {best_r2}")
