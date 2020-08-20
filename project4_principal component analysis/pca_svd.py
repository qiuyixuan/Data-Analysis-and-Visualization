'''pca_svd.py
Subclass of PCA_COV that performs PCA using the singular value decomposition (SVD)
Yixuan Qiu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np

import pca_cov


class PCA_SVD(pca_cov.PCA_COV):
    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars` using SVD

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        TODO:
        - This method should mirror that in pca_cov.py (same instance variables variables need to
        be computed).
        - There should NOT be any covariance matrix calculation here!
        - You may use np.linalg.svd to perform the singular value decomposition.
        '''
        self.A = self.data[vars].to_numpy()

        if normalize:
            self.normalized = True
            mins = np.amin(self.A, axis = 0)
            maxes = np.amax(self.A, axis = 0)
            ranges = maxes - mins
            self.orig_scales = ranges
            self.A = (self.A - mins) / ranges
        else:
            self.normalized = False
        
        # center data
        self.orig_means = np.mean(self.A, 0)
        Ac = self.A - self.orig_means
            
        # compute Ac = US(V.T)
        u, s, vh = np.linalg.svd(Ac)

        # recover eigenvalues
        self.e_vals = s**2 / (len(self.A) - 1)

        # recover eigenvectors
        self.e_vecs = vh.T

        self.vars = vars
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var) 