import time
import numpy as np


class Fisherface:
    def __init__(self, pca_energy=0.95, out_dim=30):
        self.pca_energy = pca_energy
        self.n_classes = None
        self.out_dim = out_dim
        self.mean = None
        self.W_pca = None
        self.W_lda = None
        self.train_projected = None
        self.train_labels = None

    def fit(self, X, y):
        X = X.reshape(X.shape[0], -1)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        print("Extera eigen value and vectors........")
        s_time = time.time()
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        print(f"Finish! Time consume:{time.time() - s_time} s \n")

        cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        pca_dim = np.argmax(cumulative_energy >= self.pca_energy) + 1
        self.W_pca = eigenvectors[:, :pca_dim]
        X_pca = X_centered @ self.W_pca

        self.n_classes = len(np.unique(y))
        overall_mean = np.mean(X_pca, axis=0)
        Sb = np.zeros((pca_dim, pca_dim))
        Sw = np.zeros((pca_dim, pca_dim))

        for c in np.unique(y):
            X_c = X_pca[y == c]
            mean_c = np.mean(X_c, axis=0)
            Sb += X_c.shape[0] * np.outer(mean_c - overall_mean, mean_c - overall_mean)
            Sw += (X_c - mean_c).T @ (X_c - mean_c)

        eig_vals, eig_vecs = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)
        sorted_idx_lda = np.argsort(eig_vals)[::-1]
        self.W_lda = eig_vecs[:, sorted_idx_lda]

        self.train_projected = X_pca @ self.W_lda
        self.train_labels = y

    def predict(self, X):
        X = X.reshape(X.shape[0], -1)
        X_centered = X - self.mean
        X_pca = X_centered @ self.W_pca
        X_projected = X_pca @ self.W_lda[:, :self.out_dim]

        X_sq = np.sum(X_projected ** 2, axis=1)
        train_sq = np.sum(self.train_projected[:, :self.out_dim] ** 2, axis=1)
        dist_sq = X_sq[:, np.newaxis] + train_sq - 2 * X_projected @ self.train_projected[:, :self.out_dim].T
        distances = np.sqrt(np.clip(dist_sq, 0, None))

        return self.train_labels[np.argmin(distances, axis=1)]
