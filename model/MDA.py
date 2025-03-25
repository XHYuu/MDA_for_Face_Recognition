import time
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.KNNClassifier import KNNClassifier

np.random.seed(42)


class MDA:

    def __init__(self, input_dim, output_dim, epochs, epsilon):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = [np.eye(in_dim) for in_dim in input_dim]
        self.epochs = epochs
        self.dim = len(input_dim)
        self.epsilon = epsilon
        self.knn = KNNClassifier(n_neighbors=3)

    def fit(self, images, labels):
        start = time.time()
        tensor = np.moveaxis(images, 0, -1)  # [m_0,...,m_n,N]
        for t in tqdm(range(self.epochs), "Training:"):
            U_current = self.U.copy()
            stop_flag = True
            for k in range(self.dim):
                # For all dimensions except the k-th dimension (which requires updating U_k later)
                # use the updated values for the first k-1 projections
                # and for dimensions from k+1 to self.dim use the values prior to the update (still un-updated)
                Y = self.project(tensor, exclude_dim=k)
                # calculate S_W and S_B
                S_B = self.compute_S_B(Y, labels, k)
                S_W = self.compute_S_W(Y, labels, k)
                # calculate eigenvalue and eigenvector
                eig_vals, eig_vecs = eigh(S_B, S_W + 1e-2 * np.eye(S_W.shape[0]))
                # sort the eig_val in descending order
                sorted_indices = np.argsort(eig_vals)[::-1]
                eig_vecs = eig_vecs[:, sorted_indices]

                # update U[k]
                self.U[k] = eig_vecs[:, :self.output_dim[k]] + 1e-2
                if t > 1 and (np.linalg.norm(self.U[k] - U_current[k], ord='fro') >=
                              self.input_dim[k] * self.output_dim[k] * self.epsilon):
                    stop_flag = False
            if t > 2 and stop_flag:
                break
        tensor = self.project(tensor)
        self.knn.fit(tensor, labels)
        print("Finish! Time consume:", time.time() - start)

    def project(self, tensor, exclude_dim=-1):
        for mode, u in enumerate(self.U):
            if mode != exclude_dim:
                tensor = self.mode_dot(tensor, u.T, mode)
        return tensor

    def predict(self, images):
        """
        use knn to perdict result
        """
        tensor = self.project(np.moveaxis(images, 0, -1))
        pred = self.knn.predict(tensor)
        # self.draw()
        return pred

    @staticmethod
    def mode_dot(tensor, matrix, mode):
        """
        Mode Product
        :param tensor: (m_1, m_2, ..., m_n)
        :param matrix: change the mode-th of tensor(From m to m')
        :param mode: dot dimension
        :return: (m_1, m_2, ..., m_mode-1, J, m_mode+1, ..., m_n)
        """
        # change the mode-th to 0 dim
        # and then can easily make dot product
        new_order = [mode] + [i for i in range(tensor.ndim) if i != mode]
        transposed_tensor = np.transpose(tensor, axes=new_order)

        result = np.tensordot(matrix, transposed_tensor, axes=(1, 0))

        original_order = list(range(1, mode + 1)) + [0] + list(range(mode + 1, tensor.ndim))
        tensor = np.transpose(result, axes=original_order)

        return tensor

    @staticmethod
    def compute_S_B(Y, labels, mode):
        """
        Inter-class scatter S_B
        """
        Y_total_transpose = np.moveaxis(Y, mode, 0)
        Y_in_mode = Y_total_transpose.reshape(Y_total_transpose.shape[0], -1, Y.shape[-1])
        overall_mean = np.mean(Y_in_mode, axis=-1)
        S_B = 0
        unique_labels = np.unique(labels)
        for c in unique_labels:
            class_indices = np.where(labels == c)[0]
            Y_class_transpose = np.moveaxis(Y[..., class_indices], mode, 0)
            Y_in_class_mode = Y_class_transpose.reshape(Y_class_transpose.shape[0], -1, len(class_indices))
            class_mean = np.mean(Y_in_class_mode, axis=-1)  # mean value for each class
            n_c = len(class_indices)
            diff = class_mean - overall_mean
            for j in range(class_mean.shape[1]):
                S_B += n_c * np.outer(diff[:, j], diff[:, j])
        return S_B

    @staticmethod
    def compute_S_W(Y, labels, mode):
        """
        Intra-class scatter S_W
        """
        S_W = 0
        unique_labels = np.unique(labels)
        for c in unique_labels:
            class_indices = np.where(labels == c)[0]
            Y_class_transpose = np.moveaxis(Y[..., class_indices], mode, 0)
            Y_in_class_mode = Y_class_transpose.reshape(Y_class_transpose.shape[0], -1, len(class_indices))
            class_mean = np.mean(Y_in_class_mode, axis=-1)  # mean value for each class
            for i in class_indices:
                Y_transpose = np.moveaxis(Y[..., i], mode, 0)
                Y_in_mode = Y_transpose.reshape(Y_transpose.shape[0], -1)  # reshape each data point
                diff = Y_in_mode - class_mean
                for j in range(Y_in_mode.shape[1]):
                    S_W += np.outer(diff[:, j], diff[:, j])
        return S_W