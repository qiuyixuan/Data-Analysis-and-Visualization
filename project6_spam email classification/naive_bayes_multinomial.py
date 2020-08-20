'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Yixuan Qiu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor sets the number of classes (int: num_classes) this classifier
        will be trained to detect. All other fields initialized to None.'''
        
        self.num_classes = num_classes

        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''

        self.class_priors = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            arr = i * np.ones(len(data))
            count = np.count_nonzero(y==arr)    # number of matches
            self.class_priors[i] = count / len(data)

        self.class_likelihoods = np.zeros((self.num_classes, data.shape[1]))

        for i in range(self.num_classes):
            indices = np.nonzero(y == i * np.ones(len(data)))
            count = np.sum(data[indices], axis = 0)            
            total_count = np.sum(count)
            self.class_likelihoods[i] = (count + 1) / (total_count + data.shape[1])


    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - Process test samples one-by-one.
        - Look up the likelihood (from training) ONLY AT the words that appear > 0 times in the
        current test sample.
        - Take the log and sum these likelihoods together.
        - Solve for posterior for each test sample i (see notebook for equation).
        - Predict the class of each test sample according to the class that produces the largest
        posterior probability.
        '''
        pred_classes = np.zeros(len(data))
        for i in range(len(data)):
            nonzero_indices = np.nonzero(data[i])[0]
            nonzero_likelihood = self.class_likelihoods[:, nonzero_indices]
            posterior = np.log(self.class_priors) + np.sum(np.log(nonzero_likelihood), axis = -1)
            pred_classes[i] = np.argmax(posterior)

        return pred_classes.astype(int)

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
