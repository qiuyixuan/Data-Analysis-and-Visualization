'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Yixuan Qiu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import analysis
import data
import math
import palettable.colorbrewer as colorbrewer
import matplotlib.markers as markers
from mpl_toolkits.mplot3d import Axes3D
import time


class Transformation(analysis.Analysis):

    def __init__(self, data_orig, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        data_orig: Data. shape=(N, num_vars).
            An array containing the original data array (only containing all the numeric variables
            — `num_vars` in total).
        data: Data (or None). shape=(N, num_proj_vars).
            An array containing all the samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variables for `data_orig`.
        '''
        super().__init__(data)
        self.data_orig = data_orig

    def project(self, headers):
        '''Project the data on the list of data variables specified by `headers` — i.e. select a
        subset of the variables from the original dataset. In other words, populate the instance
        variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list dictates the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables are optional.

        HINT: Update self.data with a new Data object and fill in appropriate optional parameters
        (except for `filepath`)

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables).
        - Make sure that you create 'valid' values for all the `Data` constructor optional parameters
        (except you dont need `filepath` because it is not relevant).
        '''
        h2c = dict(zip(headers, range(len(headers))))
        self.data = data.Data(headers = headers, data = self.data_orig.select_data(headers), header2col = h2c)

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''

        return np.hstack((self.data.get_all_data(), np.ones([len(self.data.get_all_data()), 1])))

    def translation_matrix(self, headers, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''

        M = len(self.data.get_headers_2())
        header_indices = self.data.get_header_indices(headers)
        translation_mat = np.eye(M + 1)
        for i in range(len(headers)):   
            translation_mat[header_indices[i], M] = magnitudes[i]

        return translation_mat

    def scale_matrix(self, headers, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''

        M = len(self.data.get_headers_2())
        header_indices = self.data.get_header_indices(headers)
        scale_mat = np.eye(M + 1)
        for i in range(len(headers)):   
            scale_mat[header_indices[i], header_indices[i]] = magnitudes[i]

        return scale_mat

    def rotation_matrix_2d(self, degrees):
        '''Make an 2-D rotation matrix for rotating the projected data.

        Parameters:
        -----------
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(2, 2). The 2D rotation matrix.
        '''

        # Convert angle from degrees to radians
        rad = math.radians(degrees)
        rot_mat = np.eye(2)
        rot_mat[0,0] = np.cos(rad)
        rot_mat[0,1] = -np.sin(rad)
        rot_mat[1,0] = np.sin(rad)
        rot_mat[1,1] = np.cos(rad)

        return rot_mat        

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data about the ONE
        axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        # Convert angle from degrees to radians
        rad = math.radians(degrees)

        rot_mat = np.eye(4)
        header_list = self.data.get_headers_2()

        # if rotate about x axis
        if header == header_list[0]:
            rot_mat[1,1] = np.cos(rad)
            rot_mat[1,2] = -np.sin(rad)
            rot_mat[2,1] = np.sin(rad)
            rot_mat[2,2] = np.cos(rad)
        # if rotate about y axis
        elif header == header_list[1]:
            rot_mat[0,0] =  np.cos(rad)
            rot_mat[2,0] = -np.sin(rad)
            rot_mat[0,2] = np.sin(rad)
            rot_mat[2,2] = np.cos(rad)
        # if rotate about z axis
        elif header == header_list[2]:
            rot_mat[0,0] =  np.cos(rad)
            rot_mat[0,1] = -np.sin(rad)
            rot_mat[1,0] = np.sin(rad)
            rot_mat[1,1] = np.cos(rad)
        else:
            print("invalid header string")

        return rot_mat

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected dataset after it has been transformed by `C`
        '''
        A = self.get_data_homogeneous()
        return (C@A.T).T

    def translate(self, headers, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        translated_data = (self.translation_matrix(headers, magnitudes)@self.get_data_homogeneous().T).T
        translated_data = translated_data[:, :-1]
        new_headers = self.data.get_headers_2()
        new_h2c = self.data.get_mappings()
        self.data = data.Data(headers=new_headers, data=translated_data, header2col=new_h2c)
        return translated_data

    def scale(self, headers, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        
        scaled_data = (self.scale_matrix(headers, magnitudes)@self.get_data_homogeneous().T).T
        scaled_data = scaled_data[:, :-1]
        new_headers = self.data.get_headers_2()
        new_h2c = self.data.get_mappings()
        self.data = data.Data(headers=new_headers, data=scaled_data, header2col=new_h2c)
        return scaled_data

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
                
        rotated_data = (self.rotation_matrix_3d(header, degrees)@self.get_data_homogeneous().T).T
        rotated_data = rotated_data[:, :-1]
        new_headers = self.data.get_headers_2()
        new_h2c = self.data.get_mappings()
        self.data = data.Data(headers=new_headers, data=rotated_data, header2col=new_h2c)
        return rotated_data

    def rotate_2d(self, degrees):
        '''Rotates the projected data by the angle (in degrees) `degrees`.

        Parameters:
        -----------
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''
                
        rotated_data = (self.rotation_matrix_2d(degrees)@self.data.get_all_data().T).T
        new_headers = self.data.get_headers_2()
        new_h2c = self.data.get_mappings()
        self.data = data.Data(headers=new_headers, data=rotated_data, header2col=new_h2c)
        return rotated_data        

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.
        Using Vectorization.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        start_time = time.time()
        min = np.amin(self.data.get_all_data())    # global min
        max = np.amax(self.data.get_all_data())    # global max
        range = max - min   # global range
        norm_mat = (self.data.get_all_data() - min) / range
        new_headers = self.data.get_headers_2()
        new_h2c = self.data.get_mappings()
        self.data = data.Data(headers=new_headers, data=norm_mat, header2col=new_h2c)
        end_time = time.time()
        print('\nNormalize together using vectorization\ntime elapsed: ', end_time - start_time)        

        return norm_mat

    def normalize_together_2(self):
        '''Normalize all variables in the projected dataset together.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        start_time = time.time()
        min = np.amin(self.data.get_all_data())    # global min
        max = np.amax(self.data.get_all_data())    # global max
        range = max - min   # global range
        mins = np.ones(len(self.data.get_headers_2())) * min
        ranges = np.ones(len(self.data.get_headers_2())) * (max - min)
        trans_mat = self.translate(self.data.get_headers_2(), -mins)
        norm_mat = self.scale(self.data.get_headers_2(), 1/ranges)   
        end_time = time.time()
        print('\nNormalize together\ntime elapsed: ', end_time - start_time)         

        return norm_mat        

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        start_time = time.time()
        mins = np.amin(self.data.get_all_data(), axis = 0)
        maxes = np.amax(self.data.get_all_data(), axis = 0)
        ranges = maxes - mins
        trans_mat = self.translate(self.data.get_headers_2(), -mins)
        norm_mat = self.scale(self.data.get_headers_2(), 1/ranges)
        end_time = time.time()
        print('\nNormalize separately\ntime elapsed: ', end_time - start_time)        

        return norm_mat

    def normalize_separately_2(self):
        '''Normalize each variable separately using numpy vectorization/broadcasting.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        start_time = time.time()
        mins = np.amin(self.data.get_all_data(), axis = 0)
        maxes = np.amax(self.data.get_all_data(), axis = 0)
        ranges = maxes - mins
        trans_mat = self.data.get_all_data() - mins
        norm_mat = trans_mat / ranges
        end_time = time.time()
        print('\nNormalize separately using vectorization\ntime elapsed: ', end_time - start_time)

        return norm_mat    

    def zscore(self):
        '''Normalize by z-score.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''

        mean = np.amin(self.data.get_all_data())    # global mean
        std = np.std(self.data.get_all_data())  # standard deviation
        means = np.ones(len(self.data.get_headers_2())) * mean
        stds = np.ones(len(self.data.get_headers_2())) * std
        norm_mat = (self.data.get_all_data() - means) / stds
        new_headers = self.data.get_headers_2()
        new_h2c = self.data.get_mappings()
        self.data = data.Data(headers=new_headers, data=norm_mat, header2col=new_h2c)      

        return norm_mat            

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Section B (Linear Algebra): Use a ColorBrewer color palette (e.g. from the
            `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        
        fig, ax = plt.subplots(figsize = [6, 4])

        x = self.data.select_data(ind_var)
        y = self.data.select_data(dep_var)
        z = self.data.select_data(c_var)

        s = ax.scatter(x, y, c=z, cmap=colorbrewer.sequential.Greys_9.mpl_colormap)
        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)
        bar = fig.colorbar(s, ax = ax)
        bar.set_label(c_var)
        ax.set_title(title)

    def scatter_color_4d(self, ind_var, dep_var, c_var, s_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension, 
            and a marker size scale representing the 4th dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
        s_var: Header of the variable that will be plotted along the marker size axis.
            NOTE: Section B (Linear Algebra): Use a ColorBrewer color palette (e.g. from the
            `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        
        fig, ax = plt.subplots(figsize = [12, 12])

        x = self.data.select_data(ind_var)
        y = self.data.select_data(dep_var)
        z = self.data.select_data(c_var)
        size = self.data.select_data(s_var)

        s = ax.scatter(x, y, size*20, c=z, cmap=colorbrewer.sequential.Greys_9.mpl_colormap)
        handles, labels = s.legend_elements(prop="sizes", alpha=0.6)
        legend2 = ax.legend(handles, labels, loc="lower right", title=s_var + ' scaled by 20')
        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)
        bar = fig.colorbar(s, ax = ax)
        bar.set_label(c_var)
        ax.set_title(title)        

    def scatter_color_5d(self, x_var, y_var, z_var, c_var, s_var, title=None):
        '''Creates a 5D scatter plot with a color scale representing the 4th dimension, 
            and a marker size scale representing the 5th dimension.

        Parameters:
        -----------
        x_var: str. Header of the variable that will be plotted along the X axis.
        y_var: Header of the variable that will be plotted along the Y axis.
        z_var: Header of the variable that will be plotted along the Z axis.
        c_var: Header of the variable that will be plotted along the color axis.
        s_var: Header of the variable that will be plotted along the marker size axis.
            NOTE: Section B (Linear Algebra): Use a ColorBrewer color palette (e.g. from the
            `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        x = self.data.select_data(x_var)
        y = self.data.select_data(y_var)
        z = self.data.select_data(z_var)
        color = self.data.select_data(c_var)
        size = self.data.select_data(s_var)  

        fig = plt.figure(figsize = [12, 12])
        ax = fig.add_subplot(111, projection='3d')
        sct = ax.scatter(x, y, z, s=size*30, c=color.flatten(), cmap=colorbrewer.sequential.Blues_9.mpl_colormap)
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_zlabel(z_var)
        handles, labels = sct.legend_elements(prop="sizes", alpha=0.6)
        legend = ax.legend(handles, labels, loc="upper left", title=s_var + ' scaled by 30')
        bar = fig.colorbar(sct, ax = ax)
        bar.set_label(c_var)
        ax.set_title(title) 


    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers_2(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots(figsize = [6, 4])
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap)

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls )
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax