import mlflow.pyfunc
import numpy as np
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        self.model = mlflow.pyfunc.load_model(
            model_uri="models:/WineQualityElasticNet/Production"
        )

    def predict(self, data):
        return self.model.predict(data)
