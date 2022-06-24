from typing import Optional
import numpy as np
from sklearn.preprocessing import normalize


class CosineCluster:
    def __init__(self, n_clusters: int, max_num_iterations: int = 20) -> None:
        self.n_clusters = n_clusters

        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_indices: Optional[np.ndarray] = None
        self.clusters_converged: bool = False

        self.max_num_iterations: int = max_num_iterations

    def _init_cluster_centers(self, input_data: np.ndarray, random_init: bool = False) -> np.ndarray:
        """ Initialize the cluster centers """

        if random_init:
            self.cluster_centers = normalize(np.random.rand(self.n_clusters, input_data.shape[1]), axis=1)
        else:
            self.cluster_centers = input_data[range(self.n_clusters)]
        self.clusters_converged: bool = False

    def _calculate_intertia(self, normalized_input_data: np.ndarray) -> float:
        n_data_points = normalized_input_data.shape[0]
        cosine_similarity = normalized_input_data.dot(self.cluster_centers.transpose())
        return n_data_points - np.sum(np.max(cosine_similarity, axis=1))

    def _update_cluster_centers(self, normalized_input_data: np.ndarray) -> np.ndarray:
        """
        Update cluster centers with normalized inputdata
        """

        # For each input vector, find the index of the closest cluster center
        cosine_similarity = normalized_input_data.dot(self.cluster_centers.transpose())
        cluster_indices = np.argmax(cosine_similarity, axis=1)

        # Check if clusters have changed
        self.clusters_converged = np.array_equal(cluster_indices, self.cluster_indices)
        self.cluster_indices = cluster_indices

        new_centers = np.zeros((self.n_clusters, normalized_input_data.shape[1]))
        for n in range(self.n_clusters):

            if np.sum(cluster_indices == n) != 0:
                new_center_unnormalized = np.sum(normalized_input_data[cluster_indices == n], axis=0)
                new_centers[n, :] = new_center_unnormalized / np.linalg.norm(new_center_unnormalized)
            else:
                new_centers[n, :] = normalized_input_data[n, :]

        self.cluster_centers = new_centers

    def fit(self, data: np.ndarray, random_init: bool = False) -> None:
        """ Fit the input data to self.n_clusters different clusters """
        data_normalized = normalize(data, axis=1)
        self._init_cluster_centers(data_normalized, random_init=random_init)
        iteration = 0
        while not self.clusters_converged:
            self._update_cluster_centers(data_normalized)

            if iteration > self.max_num_iterations:
                break
            iteration += 1

        self.inertia = self._calculate_intertia(data_normalized)
