'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Yixuan Qiu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # orig_means: ndarray. shape=(num_selected_vars,)
        #   Means of each orignal data variable
        self.orig_means = None

        # orig_scales: ndarray. shape=(num_selected_vars,)
        #   Ranges of each orignal data variable
        self.orig_scales = None    

        self.mins = None    

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        
        return np.dot((data-np.mean(data, 0)).T, (data-np.mean(data, 0))) / ((len(data)-1)) 

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        prop_var = []
        for i in range(len(e_vals)):
            prop_var.append(e_vals[i] / np.sum(e_vals))
        return prop_var

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_var = []
        sum = 0
        for i in range(len(prop_var)):
            sum += prop_var[i]
            cum_var.append(sum)
            
        return cum_var

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        self.A = self.data[vars].to_numpy()

        if normalize == True:
            self.normalized = True
            mins = np.amin(self.A, axis = 0)
            self.mins = mins
            maxes = np.amax(self.A, axis = 0)
            ranges = maxes - mins
            self.orig_scales = ranges
            self.A = (self.A - mins) / ranges
        else:
            self.normalized = False

        # center data
        self.orig_means = np.mean(self.A, 0)
        Ac = self.A - self.orig_means
            
        cov = self.covariance_matrix(Ac)

        self.e_vals, self.e_vecs = LA.eig(cov)
        self.vars = vars
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)    

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''

        if num_pcs_to_keep != None:     
            x = np.arange(1, num_pcs_to_keep+1)
        else:
            x = np.arange(1, len(self.prop_var)+1)

        y = self.compute_cum_var(self.prop_var[:num_pcs_to_keep])
        plt.plot(x, y, marker='o', markersize=5)
        plt.xlabel('top PCs included')
        plt.ylabel('cumulative variance accounted for')
        plt.title('Elbow Plot')

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        self.orig_means = np.mean(self.A, 0)
        Ac = self.A - self.orig_means
        surv_vecs = self.e_vecs[:, pcs_to_keep]
        pca_proj = Ac @ surv_vecs
        self.A_proj = pca_proj
        return pca_proj

    def loading_plot(self):
        '''Create a loading plot of the top 2 PC eigenvectors

        TODO:
        - Plot a line joining the origin (0, 0) and corresponding components of the top 2 PC eigenvectors.
            Example: If e_1 = [0.1, 0.3] and e_2 = [1.0, 2.0], you would create two lines to join
            (0, 0) and (0.1, 1.0); (0, 0) and (0.3, 2.0).
            Number of lines = num_vars
        - Use plt.annotate to label each line by the variable that it corresponds to.
        - Reminder to create useful x and y axis labels.

        NOTE: Don't write plt.show() in this method
        '''
        e_1 = self.get_eigenvectors()[:, 0]
        e_2 = self.get_eigenvectors()[:, 1]
        plt.figure(figsize=(12, 12))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Loading Plot')
        for i in range(len(e_1)):
            plt.arrow(0, 0, e_1[i], e_2[i])
            plt.annotate(self.vars[i], (e_1[i], e_2[i]), fontsize=12)

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        '''

        pcs_to_keep = np.arange(0, top_k)
        pca_proj = self.pca_project(pcs_to_keep)

        if self.normalized:
            # scale by original data range if normalized
            return (pca_proj@(self.e_vecs[:, pcs_to_keep].T) + self.orig_means) * self.orig_scales + self.mins
        else:
            return pca_proj@(self.e_vecs[:, pcs_to_keep].T) + self.orig_means