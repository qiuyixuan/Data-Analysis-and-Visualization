'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Yixuan Qiu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import math

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy'):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor), except
        for self.adj_R2.

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data(self.dep_var)

        if method == 'scipy':
            c = self.linear_regression_scipy(self.A, self.y)
        elif method == 'normal':
            c = self.linear_regression_normal(self.A, self.y)
        elif method == 'qr':
            c = self.linear_regression_qr(self.A, self.y)
        else:
            print('invalid method')
        
        self.slope = c[:-1]
        self.intercept = float(c[-1])
        y_pred = self.predict(self.slope, self.intercept)
        self.R2 = self.r_squared(y_pred)
        self.residuals = self.y - y_pred

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        x = np.hstack((A, np.ones([len(A), 1])))
        c, res, rnk, s = scipy.linalg.lstsq(x, y)
        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        A = np.hstack((A, np.ones([len(A), 1])))
        c = scipy.linalg.inv((A.T)@A)@(A.T)@y
        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''

        A1 = np.hstack([A, np.ones([len(A), 1])])
        Q, R = self.qr_decomposition(A1)
        c = scipy.linalg.solve_triangular(R, Q.T@y)
        return c

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.
        
        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''

        Q = np.ndarray((A.shape))  # initialize Q
        
        count = 0   # col index

        # cols in A = rows in A.T
        for row in A.T:
            col = np.copy(row)
            for i in range(count):
                proj = np.dot(Q[:, i], row) * Q[:, i]
                col -= proj
            Q[:, count] = col / np.linalg.norm(col)
            count += 1  # increment col index

        R = Q.T@A

        return Q, R

    def predict(self, slope, intercept, X=None):
        '''Use fitted linear regression model to predict the values of data matrix `X`.
        Generates the predictions y_pred = mD + b, where (m, b) are the model fit slope and intercept,
        D is the data matrix.

        Parameters:
        -----------
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.
        
        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if X is None:
            y_pred = self.A @ slope + intercept
        else:
            if self.p > 1:
                y_pred = self.make_polynomial_matrix(X, self.p) @ slope + intercept
            else:
                y_pred = X @ slope + intercept
            
        return y_pred

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        E = np.sum((self.y - y_pred)**2)
        S = np.sum((self.y - np.mean(self.y))**2)
        R2 = 1 - E/S
        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the 
            data samples
        '''
        res = self.y - y_pred
        return res

    def mean_sse(self, X=None):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Parameters:
        -----------
        X: ndarray. shape=(anything, num_ind_vars)
            Data to get regression predictions on.
            If None, get predictions based on data used to fit model.

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''
        y_pred = self.predict(self.slope, self.intercept, X)
        res = self.compute_residuals(y_pred)
        msse = np.sum(res**2) / len(res)

        return msse


    def scatter(self, ind_var, dep_var, title, ind_var_index=0):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.
        
        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot
        ind_var_index: int. Index of the independent variable in self.slope
            (which regression slope is the right one for the selected independent variable
            being plotted?)
            By default, assuming it is at index 0.

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        if self.p == 1:
            x_pts, y_pts = analysis.Analysis(self.data).scatter(ind_var, dep_var, title=title)
            line_x = np.linspace(np.amin(x_pts), np.amax(x_pts), 100)
            line_y = self.slope[ind_var_index] * line_x + self.intercept
            plt.plot(line_x, line_y, c='m', label = 'linear regression')
            y_pred = self.predict(self.slope, self.intercept)
            # R2 = round(self.r_squared(y_pred), 2)
            R2 = self.r_squared(y_pred)
            plt.title(title + '  R^2: ' + str(R2))

        else:   # self.p > 1
            x_pts, y_pts = analysis.Analysis(self.data).scatter(ind_var, dep_var, title=title)
            line_x = np.linspace(np.amin(x_pts), np.amax(x_pts), 100)
            # x_mat = self.make_polynomial_matrix(line_x, self.p)
            x_mat = np.ndarray((len(line_x), len(self.slope)))
            for i in range(len(self.slope)):
                x_mat[:, i] = line_x**(i+1)
            # line_y = self.slope[ind_var_index] * line_x + self.intercept
            line_y = x_mat @ self.slope + self.intercept
            plt.plot(line_x, line_y, c='m', label = 'linear regression')
            y_pred = self.predict(self.slope, self.intercept)
            R2 = round(self.r_squared(y_pred), 2)
            plt.title(title + '  R^2: ' + str(R2))

    def pair_plot(self, data_vars, fig_sz=(12, 12)):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, axes = analysis.Analysis(self.data).pair_plot(data_vars, fig_sz=fig_sz, title='Pair Plot')
        
        for row in range(len(data_vars)):
            for col in range(len(data_vars)):
                x_pts = self.data.select_data(data_vars[row])
                line_x = np.linspace(np.amin(x_pts), np.amax(x_pts), 100)
                self.linear_regression(data_vars[row], data_vars[col], method='scipy')
                line_y = self.slope[0] * line_x + self.intercept
                axes[row][col].plot(line_x, line_y, c='m')
                y_pred = self.predict(self.slope, self.intercept)
                R2 = round(self.r_squared(y_pred), 2)
                axes[row][col].set_title('R^2: ' + str(R2))

                # clear the main diagonal scatter plots and place histograms
                if (row == col):
                    axes[row][col].clear()
                    axes[row][col].hist(x_pts)

                # set x and y axes labels
                if row == len(data_vars) - 1:
                    axes[len(data_vars) - 1, col].set_xlabel(data_vars[col])
                if col == 0:    
                    axes[row, 0].set_ylabel(data_vars[row])     
               

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.
        
        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        poly_mat = np.ndarray((len(A), p))
        for i in range(0, p):
            poly_mat[:, i] = A[:, 0]**(i+1)
        return poly_mat

    def poly_regression(self, ind_var, dep_var, p, method='normal'):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        
        (Week 2)
        
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10
            (The method that you call for the linear regression solver will take care of the intercept)
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create the independent variable data matrix (self.A) with columns appropriate for
            polynomial regresssion. Do this with self.make_polynomial_matrix
            - You should programatically generate independent variable name strings based on the
            polynomial degree.
                Example: ['X_p1, X_p2, X_p3'] for a cubic polynomial model
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        self.ind_vars = []
        names = ''
        for i in range(1, p+1):
            names += ind_var + '_p' + str(i)
            if i != p:
                names += ', '
                
        self.ind_vars.append(names)
        self.dep_var = dep_var
        self.A = self.make_polynomial_matrix(self.data.select_data([ind_var]), p)
        self.y = self.data.select_data(self.dep_var)

        if method == 'scipy':
            c = self.linear_regression_scipy(self.A, self.y)
        elif method == 'normal':
            c = self.linear_regression_normal(self.A, self.y)
        elif method == 'qr':
            c = self.linear_regression_qr(self.A, self.y)
        else:
            print('invalid method')
        
        self.slope = c[:-1]
        self.intercept = float(c[-1])
        y_pred = self.predict(self.slope, self.intercept)
        self.R2 = self.r_squared(y_pred)
        self.residuals = self.y - y_pred
        self.p = p