import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)


class Eigenface:
    def __init__(self, components):
        self.pca = PCA(n_components=components)
        self.knn = KNeighborsClassifier(n_neighbors=3)

    def fit(self, train_image, labels):
        train_image = train_image.reshape(train_image.shape[0], -1)
        X_train_pca = self.pca.fit_transform(train_image)

        self.knn.fit(X_train_pca, labels)

    def predict(self, images):
        test_image = images.reshape(images.shape[0], -1)
        X_test_pca = self.pca.transform(test_image)

        return self.knn.predict(X_test_pca)
