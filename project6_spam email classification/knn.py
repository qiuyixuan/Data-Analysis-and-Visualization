'''knn.py
K-Nearest Neighbors algorithm for classification
Yixuan Qiu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import operator
from palettable import cartocolors
import scipy


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    
    def __init__(self, num_classes):
        '''KNN constructorsets the number of classes (int: num_classes) this classifier
        will be trained to detect. All other fields initialized to None.'''
        
        self.num_classes = num_classes

        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data
        self.classes = y

    def euclidean_dist(self, pt_1, pt_2):
        return np.sqrt(np.sum((pt_1 - pt_2)**2))

    def predict(self, data, k, dist_metrics='L2'):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''        
        pred_classes = np.zeros(len(data))

        for j in range(len(data)):
            distances = []
            for i in range(len(self.exemplars)):
                if dist_metrics == 'L2':
                    dist = self.euclidean_dist(data[j], self.exemplars[i])
                else:   # L1
                    dist = scipy.spatial.distance.cityblock(data[j], self.exemplars[i])     
                distances.append((dist, self.classes[i]))
            
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:k]
            
            counter = np.zeros(len(self.classes))
            for neighbor in neighbors:
                counter[int(neighbor[1])] += 1
            
            pred_classes[j] = np.argmax(counter)

        return pred_classes

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        return (len(y) - np.count_nonzero(y - y_pred)) / len(y)

    def plot_predictions(self, k, n_sample_pts, title=None):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        
        Color Palettes:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use a ColorBrewer color palette. List of possible ones here:
        https://github.com/CartoDB/CartoColor/wiki/CARTOColor-Scheme-Names
            - An example: cartocolors.qualitative.Safe_4.mpl_colors
            - The 4 stands for the number of colors in the palette. For simplicity, you can assume
            that we're hard coding this at 4 for 4 classes.
        - Each ColorBrewer palette is a Python list. Wrap this in a `ListedColormap` object so that
        matplotlib can parse it (already imported above).
        - If you can't get ColorBrewer to work, that's OK. Google Matplotlib colormaps, and pick
            one that doesn't use both red and green in the first 4 discrete colors.
        
        The Rest:
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        cmap = ListedColormap(cartocolors.qualitative.Safe_4.mpl_colors)
        samp_vec = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(samp_vec, samp_vec)
        data = np.column_stack((x.flatten(), y.flatten())).reshape((n_sample_pts**2, 2))
        y_pred = self.predict(data, k).reshape((n_sample_pts, n_sample_pts))
        im = plt.pcolormesh(x, y, y_pred, cmap=cmap)
        plt.title(title)
        plt.colorbar(im)
        plt.show()

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        cm = np.zeros((self.num_classes, self.num_classes))
        cm[0, 0] = len(y) - np.count_nonzero(y)
        cm[1, 1] = np.count_nonzero(y)
        diff = y - y_pred
        false_inds = np.nonzero(diff)[0]    # indices of wrong predictions
       
        # 0 - 0 = 0  spam -> spam
        # 0 - 1 = - 1  spam -> ham  x
        # 1 - 0 = 1  ham -> spam  x
        # 1 - 1 = 0  ham -> ham
        
        for idx in false_inds:
            if diff[idx] == -1:
                cm[0, 1] += 1
                cm[0, 0] -= 1
            elif diff[idx] == 1:
                cm[1, 0] += 1
                cm[1, 1] -= 1
                
        return cm.astype(int)
