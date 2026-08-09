import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler


class ClassicalAnomalyDetector:
    """Classical anomaly detector using feature normalization and robust covariance."""

    def __init__(self, contamination: float = 0.05):
        self.scaler = StandardScaler()

        self.model = EllipticEnvelope(
            contamination=contamination,
            random_state=42,
        )

    def fit(self, features: np.ndarray) -> None:
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)

    def score(self, features: np.ndarray) -> np.ndarray:
        scaled_features = self.scaler.transform(features)

        # Higher values should correspond to more anomalous samples.
        return -self.model.decision_function(scaled_features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.score(features)

        threshold = np.median(scores)

        return (scores >= threshold).astype(int)