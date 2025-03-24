from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class LDAProcessor:
    def __init__(self, n_components):
        self.lda = LDA(n_components=n_components)
        self.knn = KNeighborsClassifier(n_neighbors=3)  # Placeholder for KNN model

    def fit(self, X_train, y_train):
        """
        Fit the LDA model to the data.

        :param X: Feature matrix
        :param y: Target labels
        """
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_train_transformed = self.lda.fit_transform(X_train, y_train)
        self.knn.fit(X_train_transformed, y_train)

    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0], -1)

        X_test_transformed = self.lda.transform(X_test)

        # Predict and return results
        return self.knn.predict(X_test_transformed)
