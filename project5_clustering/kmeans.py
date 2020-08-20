'''kmeans.py
Performs K-Means clustering
Yixuan Qiu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable.colorbrewer as colorbrewer


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.sqrt(np.sum((pt_1 - pt_2)**2))

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.sqrt(np.sum((pt - centroids)**2, axis = 1))

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        # row indices of centroids. shape: (k, )
        r_idx = np.random.choice(len(self.data), k, replace=False)
        return np.reshape(self.data[r_idx], (k, self.num_features))

    def initialize_plusplus(self, k):
        '''Initializes K-means by setting the initial centroids (means) according to the K-means++
        algorithm

        (LA section only)

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        TODO:
        - Set initial centroid (i = 0) to a random data sample.
        - To pick the i-th centroid (i > 0)
            - Compute the distance between all data samples and i-1 centroids already initialized.
            - Create the distance-based probability distribution (see notebook for equation).
            - Select the i-th centroid by randomly choosing a data sample according to the probability
            distribution.
        '''
        centroids = np.zeros((k, self.num_features))
        c1_idx = np.random.choice(self.num_samps)
        centroids[0] = self.data[c1_idx]

        if k > 1:
            for j in range(1, k):
                dists = np.zeros(self.num_samps)
                for i in range(self.num_samps):
                    min_dist = np.amin(self.dist_pt_to_centroids(self.data[i], centroids[:j]))
                    dists[i] = min_dist
                
                # p = dists**2 / np.sum(dists**2)
                p = dists / np.sum(dists)
                next_idx = np.random.choice(self.num_samps, p = p)
                centroids[j] = self.data[next_idx]

        return np.reshape(centroids, (k, self.num_features))

    def cluster(self, k=2, tol=1e-5, max_iter=1000, init_method='random', verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the difference between all the centroid values from the
        previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the difference between
        the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        self.k = k    # number of clusters
        num_iter = 0    # number of iterations
        diff = np.inf    # difference between all centroid values from the previous and current time step
        
        # initialize k cluster centroids
        if init_method == 'kmeans++':
            self.centroids = self.initialize_plusplus(k)
        else:   # init_method == 'random'
            self.centroids = self.initialize(k)

        while (num_iter < max_iter and  np.all(diff) > tol):
            # assign every data sample to closest centroid
            self.data_centroid_labels = self.update_labels(self.centroids)
            # given cluster assignments, update centroid of each cluster
            new_centroids, diff = self.update_centroids(k, self.data_centroid_labels, self.centroids)
            self.centroids = new_centroids 
            num_iter += 1
            
        self.inertia = self.compute_inertia()
        return self.inertia, num_iter

    def cluster_batch(self, k=2, n_iter=1, init_method='random', verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the difference between all the centroid values from the
        previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.
        '''
        if init_method == 'kmeans++':
            self.centroids = self.initialize_plusplus(k)
        else:
            self.centroids = self.initialize(k)

        inertia_list = []
        centroid_list = []
        label_list = []
        num_iter_list = []
        for i in range(n_iter):
            inertia, num_iter = self.cluster(k=k, init_method=init_method, verbose=verbose)
            inertia_list.append(inertia)
            centroid_list.append(self.centroids)
            label_list.append(self.data_centroid_labels)
            num_iter_list.append(num_iter)
        
        idx = np.argmin(inertia_list)
        self.centroids = centroid_list[idx]
        self.data_centroid_labels = label_list[idx]
        self.inertia = inertia_list[idx]

        return np.mean(num_iter_list)

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray. shape=(self.num_samps,). Holds index of the assigned cluster of each data sample

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = np.zeros(self.num_samps)
        for i in range(self.num_samps):     
            dists = self.dist_pt_to_centroids(self.data[i], centroids)         
            labels[i] = np.argmin(dists)
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''
        new_centroids = np.ndarray((k, self.num_features))
        
        for i in range(k):
            samples = []
            for j in range(self.num_samps):
                if data_centroid_labels[j] == i:
                    samples.append(self.data[j])

            new_centroids[i] = np.mean(np.asarray(samples), axis=0)

        centroid_diff = new_centroids - prev_centroids 

        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        sum = 0
        for i in range (self.num_samps):
            centroid_idx = int(self.data_centroid_labels[i])
            centroid = self.centroids[centroid_idx]
            dist = self.dist_pt_to_pt(self.data[i], centroid)
            sum += dist**2

        return sum / self.num_samps

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.


        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            (LA Section): You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        plt.scatter(self.data[:, 0], self.data[:, 1], c = self.data_centroid_labels, cmap=colorbrewer.qualitative.Paired_10.mpl_colormap)
        plt.plot(self.centroids[:, 0], self.centroids[:, 1], 'k+')
        plt.title('Clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def elbow_plot(self, max_k):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k-1.

        TODO:
        - Run k-means with k=1,2,...,max_k-1, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertias = []   # list to store inertias

        for i in range(1, max_k):
            inertia, num_iter = self.cluster(i)
            inertias.append(inertia) 
            
        plt.plot(np.arange(1, max_k), inertias)
        plt.xlabel('k clusters')
        plt.ylabel('inertia')
        plt.xticks(np.arange(1, max_k))
        plt.show()

    def replace_color_with_centroid(self, k=3):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        self.cluster(k=k, tol=1e-5, max_iter=1000, init_method='random', verbose=False)
        new_data = np.zeros((self.num_samps, self.num_features))
        for i in range(self.num_samps):
            label = int(self.data_centroid_labels[i])
            new_data[i] = self.centroids[label]
            self.set_data(new_data)
