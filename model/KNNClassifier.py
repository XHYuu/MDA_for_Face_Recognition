import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        M = X.shape[-1]
        predictions = []

        for i in range(M):
            x = X[..., i]
            # Calculate Frobenius distance
            distances = np.linalg.norm(self.X_train - x[..., np.newaxis], axis=tuple(range(X.ndim - 1)))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            most_common_label = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)

        return np.array(predictions)
