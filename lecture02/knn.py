import numpy as np 
import math 

class KNearestNeighbors:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        'Train' the classifier by storing the training data.

        Parameters
        ----------
        X : np.ndarray, shape (num_train, D)
            Training data. Each row is a flattened example of dimension D.
        y : np.ndarray, shape (num_train,)
            Training labels. y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def compute_l2_distance(self, x):
        """
        Compute L2 distance from a single test point x to all training points.

        x: (D,)
        returns: dists of shape (num_train,)
        """
        dists = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        return dists

    def predict(self, X, k=1):
        """
        Predict labels for test data.

        X: (num_test, D) or (D,)
        k: number of neighbors

        Returns:
            y_pred: (num_test,) or scalar if a single example was passed.
        """
        # Remember if we got a single example
        single_input = (X.ndim == 1)

        # Ensure X is 2D: (num_test, D)
        X = np.atleast_2d(X)
        num_test = X.shape[0]

        # Create empty array -- better than filling it with zeros
        y_pred = np.empty(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            # 1. distances from X[i] to all training points
            dists = self.compute_l2_distance(X[i])   # (num_train,)

            # 2. indices of k smallest distances in order
            nearest_idxs = np.argsort(dists)[:k]     # (k,)

            # 3. labels of those neighbors
            closest_y = self.y_train[nearest_idxs]   # (k,)

            # 4. majority vote
            values, counts = np.unique(closest_y, return_counts=True)
            y_pred[i] = values[np.argmax(counts)]

        # If user passed a single example, return a scalar
        if single_input:
            return y_pred[0]
        return y_pred

