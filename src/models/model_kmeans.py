"""
Name:           model_kmeans
Author:         Pauline Martin
Description:    Implementation of the KMeans algorithm.
"""
# Standard imports
import math
import numpy as np

# Internal imports
from src.models.exceptions import NotFittedError

###################
# Exposed classes #
###################


class KMeansModel:
    """
    Implementation of the K Means clustering method.

    Attributes
    ----------
    k: int
        Number of clusters to create
    max_nb_iterations: int
        Maximal number of iterations

    Methods
    -------
    trained()
        Is the ClassifierChainModel trained?
    fit(df)
        Compute the model from the dataframe.
    predict(df)
        Predict the clusters for the dataframe.
    fit_predict(df)
        Compute the model and then predict the clusters for the dataframe.
    """

    #################
    # Init function #
    #################

    def __init__(self, k=2, max_nb_iterations=100):
        """
        Init function for KMeans.

        Parameters
        ----------
        k: int, optional
            Number of clusters to create, should be superior to 1, default is 2
        max_nb_iterations: int, optional
            Maximal number of iterations, default is 100
        """
        if not isinstance(k, int) or k <= 1:
            raise TypeError(f"k should be an int superior to 1, got a {type(k)} ({k})")
        if not isinstance(max_nb_iterations, int) or max_nb_iterations < 1:
            raise TypeError(f"max_nb_iterations should be a positive int, got a {type(max_nb_iterations)}\
                ({max_nb_iterations})")
        self.k = k
        self.max_nb_iterations = max_nb_iterations

        self._centroids = []
        self._is_trained = False

    #####################
    # Private functions #
    #####################

    def _get_random_centroids(self, df):
        """
        Generate random centroids within the scope of the input data.
        """
        # Take k different random point from the dataframe
        return df.drop_duplicates().sample(self.k).values.tolist()

    def _distance(self, point1, point2):
        """
        Calculate the distance between 2 points.
        """
        res = 0
        for value1, value2 in zip(point1, point2):
            res += math.pow(value1 - value2, 2)
        return math.sqrt(res)

    def _lists_almost_equals(self, list_of_lists1, list_of_lists2, epsilon=0.0001):
        for list1, list2 in zip(list_of_lists1, list_of_lists2):
            for value1, value2 in zip(list1, list2):
                if value1 > value2 and value1 - value2 > epsilon:
                    return False
                if value1 < value2 and value2 - value1 > epsilon:
                    return False
        return True

    def _calculate_centroid(self, cluster):
        centroid = [0] * len(cluster[0])
        for point in cluster:
            for i in range(len(point)):
                centroid[i] += point[i]
        return [value / len(cluster[0]) for value in centroid]

    def _calculate_centroids(self, clusters):
        centroids = []
        for cluster in clusters:
            centroids.append(self._calculate_centroid(cluster))
        return centroids

    #####################
    # Exposed functions #
    #####################

    def trained(self):
        """
        Is the ClassifierChainModel trained?
        """
        return self._is_trained

    def fit(self, df):
        """
        Compute the model from the dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            training set that have at least one variable
        """
        X = df.copy()
        nb_iteration = self.max_nb_iterations
        # Init the centroids
        new_centroids = self._get_random_centroids(X)
        centroids = self._get_random_centroids(X)

        # Iterate until the centroids don't change
        while (
            not self._lists_almost_equals(centroids, new_centroids)
            and nb_iteration != 0
        ):
            print(nb_iteration)
            centroids = new_centroids
            nb_iteration -= 1
            # Link every point to the closest centroid
            clusters = [[] for _ in range(self.k)]
            for point in X.values.tolist():
                distance_to_centroids = [self._distance(c, point) for c in centroids]
                clusters[np.argmin(distance_to_centroids)].append(point)
            # Update the centroids
            new_centroids = self._calculate_centroids(clusters)

        self._centroids = new_centroids
        self._is_trained = True

    def predict(self, df):
        """
        Predict the clusters for the dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            data to predict

        Returns
        ----------
        list(int)
            predicted cluster
        """
        # Check that the model is trained
        if not self.trained():
            raise NotFittedError(
                "This KMeansModel instance is not fitted yet. Call 'fit' with appropriate arguments before\
                using self estimator."
            )

        # Link every sample to a cluster
        predicted_clusters = [] * len(df)
        for point in df.values.tolist():
            distance_to_centroids = [self._distance(c, point) for c in self._centroids]
            predicted_clusters.append(np.argmin(distance_to_centroids))

        return predicted_clusters

    def fit_predict(self, df):
        """
        Compute the model and then predict the clusters for the dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            training set that have at least one variable

        Returns
        ----------
        list(int)
            predicted cluster
        """
        self.fit(df)
        return self.predict(df)
