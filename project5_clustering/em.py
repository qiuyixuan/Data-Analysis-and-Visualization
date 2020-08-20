'''em.py
Cluster data using the Expectation-Maximization (EM) algorithm with Gaussians
Yixuan Qiu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.colors import LogNorm
from scipy.special import logsumexp
from IPython.display import display, clear_output


class EM():
    def __init__(self, data=None):
        '''EM object constructor.
        See docstrings of individual methods for what these variables mean / their shapes

        (Should not require any changes)
        '''
        self.k = None
        self.centroids = None
        self.cov_mats = None
        self.responsibilities = None
        self.pi = None
        self.data_centroid_labels = None

        self.loglikelihood_hist = None

        self.data = data
        self.num_samps = None
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def gaussian(self, pts, mean, sigma):
        '''(LA section)
        Evaluates a multivariate Gaussian distribution described by
        mean `mean` and covariance matrix `sigma` at the (x, y) points `pts`

        Parameters:
        -----------
        pts: ndarray. shape=(num_samps, num_features).
            Data samples at which we want to evaluate the Gaussian
            Example for 2D: shape=(num_samps, 2)
        mean: ndarray. shape=(num_features,)
            Mean of Gaussian (i.e. mean of one cluster). Same dimensionality as data
            Example for 2D: shape=(2,) for (x, y)
        sigma: ndarray. shape=(num_features, num_features)
            Covariance matrix of a Gaussian (i.e. covariance of one cluster).
            Example for 2D: shape=(2,2). For standard deviations (sigma_x, sigma_y) and constant c,
                Covariance matrix: [[sigma_x**2, c*sigma_x*sigma_y],
                                    [c*sigma_x*sigma_y, sigma_y**2]]

        Returns:
        -----------
        ndarray. shape=(num_samps,)
            Multivariate gaussian evaluated at the data samples `pts`
        '''   
        N, M = pts.shape
        arr = np.zeros(N)   
        denominator = ((2*np.pi)**(M/2))*(np.sqrt(np.linalg.det(sigma)))
        coefficient = 1 / denominator
        for i in range(N):
            prod = (pts[i]-mean).T@(np.linalg.inv(sigma))@(pts[i]-mean)
            exponent = np.exp((-1/2) * prod)
            gaussian = coefficient * exponent
            arr[i] = gaussian
        return arr

    def gaussian_scipy(self, pts, mean, sigma):
        '''(Non-LA section)
        Evaluates a multivariate Gaussian distribution described by
        mean `mean` and covariance matrix `sigma` at the (x, y) points `pts` using scipy.stats

        Parameters:
        -----------
        pts: ndarray. shape=(num_samps, num_features).
            Data samples at which we want to evaluate the Gaussian
            Example for 2D: shape=(num_samps, 2)
        mean: ndarray. shape=(num_features,)
            Mean of Gaussian (i.e. mean of one cluster). Same dimensionality as data
            Example for 2D: shape=(2,) for (x, y)
        sigma: ndarray. shape=(num_features, num_features)
            Covariance matrix of a Gaussian (i.e. covariance of one cluster).
            Example for 2D: shape=(2,2). For standard deviations (sigma_x, sigma_y) and constant c,
                Covariance matrix: [[sigma_x**2, c*sigma_x*sigma_y],
                                    [c*sigma_x*sigma_y, sigma_y**2]]

        Returns:
        -----------
        ndarray. shape=(num_samps,)
            Multivariate gaussian evaluated at the data samples `pts`
        '''
        pass

    def initalize(self, k, plusplus = False):
        '''Initialize all variables used in the EM algorithm.

        Parameters:
        -----------
        k: int. Number of clusters.

        Returns
        -----------
        None

        TODO:
        - Set k as an instance variable.
        - Initialize the log likelihood history to an empty Python list.
        - Initialize the centroids to random data samples
            shape=(k, num_features)
        - Initialize the covariance matrices to the identity matrix
        (1s along main diagonal, 0s elsewhere)
            shape=(k, num_features, num_features)
        - Initialize the responsibilities to an ndarray of 1s.
            shape=(k, num_samps)
        - Initialize the pi array (proportion of points assigned to each cluster) so that each cluster
        is equally likely.
            shape=(k,)
        '''
        self.k = k
        self.loglikelihood_hist = []

        # initialize centroids randomly
        if plusplus == False:
            rand_inds = np.random.randint(low=0, high=self.num_samps, size=(k,))       
            self.centroids = self.data[rand_inds]
        else:
            # EM version of k-means++ initialization
            self.centroids = self.initialize_plusplus(k)

        self.cov_mats = np.zeros((k, self.num_features, self.num_features))
        for i in range(k):
            self.cov_mats[i] = np.eye(self.num_features)

        self.responsibilities = (1/k) * np.ones((k, self.num_samps))

        self.pi = np.zeros(k)
        for i in range(k):
            self.pi[i] = 1/k

    def e_step(self):
        '''Expectation (E) step in the EM algorithm.
        Set self.responsibilities, the probability that each data point belongs to each of the k clusters.
        i.e. leverages the Gaussian distribution.

        NOTE: Make sure that you normalize so that the probability that each data sample belongs
        to any cluster equals 1.

        Parameters:
        -----------
        None

        Returns
        -----------
        self.responsibilities: ndarray. shape=(k, num_samps)
            The probability that each data point belongs to each of the k clusters.
        '''

        self.responsibilities = np.zeros((self.k, self.num_samps))

        for i in range(self.k):
            pi = self.pi[i]
            G = self.gaussian(self.data, self.centroids[i], self.cov_mats[i])
            self.responsibilities[i] = pi * G

        self.responsibilities /= np.sum(self.responsibilities, axis = 0, keepdims = True)  

        return self.responsibilities

    def m_step(self):
        '''Maximization (M) step in the EM algorithm.
        Set self.centroids, self.cov_mats, and self.pi, the parameters that define each Gaussian
        cluster center and spread, as well as the degree to which data points "belong" to each cluster

        TODO:
        - Compute the proportion of data points that belong to each cluster.
        - Compute the mean of each cluster. This is the mean over all data points, but weighting
        the data by the probability that they belong to that cluster.
        - Compute the covariance matrix of each cluster. Use the usual equation (for all the data),
        but before summing across data samples, make sure to weight each data samples by the
        probability that they belong to that cluster.

        NOTE: When computing the covariance matrix, use the updated cluster centroids for
        the CURRENT time step.

        Parameters:
        -----------
        None

        Returns
        -----------
        self.centroids: ndarray. shape=(k, num_features)
            Mean of each of the k Gaussian clusters
        self.cov_mats: ndarray. shape=(k, num_features, num_features)
            Covariance matrix of each of the k Gaussian clusters
            Example of a covariance matrix for a single cluster (2D data): [[1, 0.2], [0.2, 1]]
        self.pi: ndarray. shape=(k,)
            Proportion of data points belonging to each cluster.
        '''
        self.pi = self.responsibilities.mean(1)

        self.centroids = np.dot(self.responsibilities, self.data) / np.sum(self.responsibilities, axis = 1, keepdims = True)

        for i in range(self.k):
            x = self.data - self.centroids[i]
            prod = self.responsibilities[i] * x.T @ x
            self.cov_mats[i] = prod / np.sum(self.responsibilities, axis = 1)[i]
                
        return self.centroids, self.cov_mats, self.pi

    def log_likelihood(self):
        '''Compute the sum of the log of the Gaussian probability of each data sample in each cluster
        Used to determine whether the EM algorithm is converging.

        Parameters:
        -----------
        None

        Returns
        -----------
        float. Summed log-likelihood of all data samples

        NOTE: Remember to weight each cluster's Gaussian probabilities by the proportion of data
        samples that belong to each cluster (pi).
        '''
        result = np.zeros((self.k, self.num_samps))
        for i in range(self.k):
            pi_c = self.pi[i]
            f_c = self.gaussian(self.data, self.centroids[i], self.cov_mats[i])
            result[i] = pi_c * f_c
        
        log = np.log(np.sum(result, 0))
        
        return np.sum(log)

    def cluster(self, k, max_iter=100, stop_tol=1e-3, verbose=False, animate=False, plusplus=False):
        '''Main method used to cluster data using the EM algorithm
        Perform E and M steps until the change in the loglikelihood from last step to the current
        step <= `stop_tol` OR we reach the maximum number of allowed iterations (`max_iter`).

        Parameters:
        -----------
        k: int. Number of clusters.
        max_iter: int. Max number of iterations to allow the EM algorithm to run.
        stop_tol: float. Stop running the EM algorithm if the change of the loglikelihood from the
        previous to current step <= `stop_tol`.
        verbose: boolean. If true, print out the current iteration, current log likelihood,
            and any other helpful information useful for debugging.

        Returns:
        -----------
        self.loglikelihood_hist: Python list. The log likelihood at each iteration of the EM algorithm.

        NOTE: Reminder to initialize all the variables before running the EM algorithm main loop.
            (Use the method that you wrote to do this)
        NOTE: At the end, print out the total number of iterations that the EM algorithm was run for.
        NOTE: The log likelihood is a NEGATIVE float, and should increase (approach 0) if things are
            working well.
        '''
        if plusplus:
            self.initalize(k, plusplus=True)
        else:
            self.initalize(k, plusplus=False)

        num_iter = 0    # number of iterations
        diff = np.inf    #  change in the loglikelihood

        while (num_iter < max_iter and diff > stop_tol):
            prev_ll = self.log_likelihood()                
            self.e_step()
            self.m_step()
            curr_ll = self.log_likelihood()
            self.loglikelihood_hist.append(curr_ll)
            diff = curr_ll - prev_ll
            num_iter += 1

            if animate:
                clear_output(wait=True)
                self.plot_clusters(self.data)
                plt.pause(0.1)
        
            if verbose:
                print("current iteration:", num_iter)
                print("current log likelihood:", curr_ll)

        # populate self.data_centroid_labels
        self.data_centroid_labels = np.zeros(self.num_samps)
        for i in range(self.num_samps):
            self.data_centroid_labels[i] = np.argmax(self.responsibilities[:, i])

        print("total number of iterations:", num_iter)
        return self.loglikelihood_hist

    def find_outliers(self, thres=0.05):
        '''Find outliers in a dataset using clustering by EM algorithm

        Parameters:
        -----------
        thres: float. Value >= 0
            Outlier defined as data samples assigned to a cluster with probability of belonging to
            that cluster < thres

        Returns:
        -----------
        Python lists of ndarrays. len(Python list) = len(cluster_inds).
            Example if k = 2: [(array([ 0, 17]),), (array([20, 26]),)]
                The Python list has 2 entries. Each entry is a ndarray.
            Within each ndarray, indices of `self.data` of detected outliers according to that cluster.
                For above example: data samples with indices 20 and 26 are outliers according to
                cluster 2.
        '''
        outliers = []

        for i in range(self.k):
            outliers_c = [] # outliers of each cluster
            G = self.gaussian(self.data, self.centroids[i], self.cov_mats[i])
            for j in range(self.num_samps):
                # if probability < threshold and data sample belongs to cluster i
                if G[j] < thres and self.data_centroid_labels[j] == i:
                    outliers_c.append(j)

            outliers.append(np.asarray(outliers_c))

        return outliers

    def estimate_log_probs(self, xy_points):
        '''Used for plotting the clusters.

        (Should not require any changes)
        '''
        probs = np.zeros([self.k, len(xy_points)])
        for c in range(self.k):
            probs[c] = np.log(self.gaussian(xy_points, self.centroids[c], self.cov_mats[c]))
        probs += np.log(self.pi[:, np.newaxis])
        return -logsumexp(probs, axis=0)

    def get_sample_points(self, data, res):
        '''Used for plotting the clusters.

        (Should not require any changes)
        '''
        data_min = np.min(data, axis=0) - 0.5
        data_max = np.max(data, axis=0) + 0.5
        x_samps, y_samps = np.meshgrid(np.linspace(data_min[0], data_max[0], res),
                                       np.linspace(data_min[1], data_max[1], res))
        plt_samps_xy = np.c_[x_samps.ravel(), y_samps.ravel()]
        return plt_samps_xy, x_samps, y_samps

    def plot_clusters(self, data, res=100, show=True, xlabel=None, ylabel=None, title=None):
        '''Method to call to plot the clustering of `data` using the EM algorithm

        (Should not require any changes)
        '''
        # Plot points assigned to each cluster in a different color
        cluster_hard_assignment = np.argmax(self.responsibilities, axis=0)
        for c in range(self.k):
            curr_clust = data[cluster_hard_assignment == c]
            plt.plot(curr_clust[:, 0], curr_clust[:, 1], '.', markersize=7)

        # Plot centroids of each cluster
        plt.plot(self.centroids[:, 0], self.centroids[:, 1], '+k', markersize=12)

        # Get grid of (x,y) points to sample the Gaussian clusters
        xy_points, x_samps, y_samps = self.get_sample_points(data, res=res)

        # Evaluate the sample points at each cluster Gaussian. For visualization, take max prob
        # value of the clusters at each point
        probs = np.zeros([self.k, len(xy_points)])
        for c in range(self.k):
            probs[c] = self.gaussian(xy_points, self.centroids[c], self.cov_mats[c])
        probs /= probs.max(axis=1, keepdims=True)
        probs = probs.sum(axis=0)
        probs = np.reshape(probs, [res, res])

        # Make heatmap for cluster probabilities
        plt.contourf(x_samps, y_samps, probs, cmap='viridis')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if show:
            plt.show()

    def elbow_plot(self, max_k):
        ll_list = []   # list to store log likelihood

        for i in range(1, max_k):
            final_ll = self.cluster(i)[-1]
            ll_list.append(final_ll) 

        plt.plot(np.arange(1, max_k), ll_list)
        plt.xlabel('k clusters')
        plt.ylabel('log likelihood')
        plt.title('Elbow Plot')
        plt.xticks(np.arange(1, max_k))
        plt.show()

    def initialize_plusplus(self, k):
        centroids = np.zeros((k, self.num_features))
        c1_idx = np.random.choice(self.num_samps)
        centroids[0] = self.data[c1_idx]

        if k > 1:
            for j in range(1, k):
                dists = np.zeros(self.num_samps)
                for i in range(self.num_samps):
                    min_dist = np.amin(self.dist_pt_to_centroids(self.data[i], centroids[:j]))
                    dists[i] = min_dist
                
                p = dists / np.sum(dists)
                next_idx = np.random.choice(self.num_samps, p = p)
                centroids[j] = self.data[next_idx]

        return np.reshape(centroids, (k, self.num_features))

    def dist_pt_to_centroids(self, pt, centroids):
        return np.sqrt(np.sum((pt - centroids)**2, axis = 1))