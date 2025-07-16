import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should randomly sample `k` different pixels from the input
    image as the initial centroids for the K-means algorithm.
    The selected `k` pixels should be sampled uniformly from all sets
    of `k` pixels in the image.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    num_pixels = X.shape[0]
    random_indices = np.random.choice(num_pixels, k, replace=False)
    return X[random_indices]

def l_p_dist_from_centroids(X, centroids, p=2):
    '''
    Inputs:
    A single image of shape (num_pixels, 3)
    The centroids of shape (k, 3)
    The parameter p for the L_p norm distance measure.

    Output: numpy array of shape `(k, num_pixels)`,
    in which entry [j,i] holds the distance of the i-th pixel from the j-th centroid.
    '''
    # Vectorized computation using broadcasting
    # X: (num_pixels, 3), centroids: (k, 3)
    # X[:, None, :] -> (num_pixels, 1, 3)
    # centroids[None, :, :] -> (1, k, 3)
    # diff -> (num_pixels, k, 3)
    diff = np.abs(X[:, None, :] - centroids[None, :, :])
    distances = np.sum(diff ** p, axis=2) ** (1 / p)
    return distances.T  # Transpose to get (k, num_pixels)

def kmeans(X, k, p, max_iter=100, epsilon=1e-8):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the L_p distance measure.
    - max_iter: the maximum number of iterations to perform.
    - epsilon: the threshold for convergence.

    Outputs:
    - The final centroids as a numpy array.
    - The final assignment of all pixels to the closest centroids as a numpy array.
    - The final WCS as a float.
    """
    centroids = get_random_centroids(X, k)
    cluster_assignments = np.zeros(X.shape[0], dtype=int)

    for iteration in range(max_iter):
        prev_centroids = centroids.copy()

        distances = l_p_dist_from_centroids(X, centroids, p)
        cluster_assignments = np.argmin(distances, axis=0)

        # Vectorized centroid update using bincount for grouping
        new_centroids = np.zeros((k, 3))
        for j in range(k):
            mask = cluster_assignments == j
            if np.any(mask):
                new_centroids[j] = np.mean(X[mask], axis=0)
            else:
                new_centroids[j] = centroids[j]

        centroids = new_centroids

        # More efficient convergence check
        if np.sum(np.abs(centroids - prev_centroids)) < epsilon:
            break

    # Vectorized WCS calculation
    WCS = 0.0
    for j in range(k):
        mask = cluster_assignments == j
        if np.any(mask):
            cluster_pixels = X[mask]
            diff = np.abs(cluster_pixels - centroids[j])
            distances_to_centroid = np.sum(diff ** p, axis=1) ** (1 / p)
            WCS += np.sum(distances_to_centroid ** p)

    return centroids, cluster_assignments, WCS

def kmeans_pp(X, k, p, max_iter=100, epsilon=1e-8):
    """
    The kmeans algorithm with alternative centroid initalization.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the L_p distance measure.
    - max_iter: the maximum number of iterations to perform.
    - epsilon: the threshold for convergence.

    Outputs:
    - The final centroids as a numpy array.
    - The final assignment of all pixels to the closest centroids as a numpy array.
    - The final WCS as a float.
    """
    num_pixels = X.shape[0]
    centroids = np.zeros((k, 3))

    # Use get_random_centroids for first centroid selection
    centroids[0] = get_random_centroids(X, 1)[0]

    for j in range(1, k):
        # Vectorized distance calculation to existing centroids
        existing_centroids = centroids[:j]
        diff = np.abs(X[:, None, :] - existing_centroids[None, :, :])
        distances_to_existing = np.sum(diff ** p, axis=2) ** (1 / p)
        distances_to_closest = np.min(distances_to_existing, axis=1)

        squared_distances = distances_to_closest ** 2
        probabilities = squared_distances / np.sum(squared_distances)

        next_centroid_idx = np.random.choice(num_pixels, p=probabilities)
        centroids[j] = X[next_centroid_idx]

    # Use standard kmeans iterations with initialized centroids
    cluster_assignments = np.zeros(num_pixels, dtype=int)

    for iteration in range(max_iter):
        prev_centroids = centroids.copy()

        distances = l_p_dist_from_centroids(X, centroids, p)
        cluster_assignments = np.argmin(distances, axis=0)

        new_centroids = np.zeros((k, 3))
        for j in range(k):
            mask = cluster_assignments == j
            if np.any(mask):
                new_centroids[j] = np.mean(X[mask], axis=0)
            else:
                new_centroids[j] = centroids[j]

        centroids = new_centroids

        if np.sum(np.abs(centroids - prev_centroids)) < epsilon:
            break

    # Vectorized WCS calculation
    WCS = 0.0
    for j in range(k):
        mask = cluster_assignments == j
        if np.any(mask):
            cluster_pixels = X[mask]
            diff = np.abs(cluster_pixels - centroids[j])
            distances_to_centroid = np.sum(diff ** p, axis=1) ** (1 / p)
            WCS += np.sum(distances_to_centroid ** p)

    return centroids, cluster_assignments, WCS