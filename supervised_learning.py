import matplotlib.pyplot as plt
from matplotlib import lines
import numpy as np
import binarytree
import scipy.stats  # Used for "mode" - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
import random as rnd

# The code below is "starter code" for graded assignment 2 in DTE-2602
# You should implement every method / function which only contains "pass".
# "Helper functions" can be left unedited. 
# Feel free to add additional classes / methods / functions to answer the assignment.
# You can use the modules imported above, and you can also import any module included 
# in Python 3.10. See https://docs.python.org/3.10/py-modindex.html .
# Using any other modules / external libraries is NOT ALLOWED.


#########################################
#   Data input / prediction evaluation
#########################################

def read_data():
    """ Read Palmer penguin data from CSV file and remove rows with missing data 
    
    # Returns:
    X:  Numpy array, shape (n_samples,4), where n_samples is number of rows 
        in the dataset. Contains the four numeric columns in the dataset 
        (bill length, bill depth, flipper length, body mass).
        Each column (each feature) is normalized by subtracting the column mean 
        and dividing by the column std.dev. ("z-score")
    y:  Numpy array, shape (n_samples,) 
        Contains integer value representing the penguin species, encoded as follows
        (alphabetically according to species name):
            'Adelie':       0
            'Chinstrap':    1
            'Gentoo':       2

    # Notes:
    - Z-score normalization: https://en.wikipedia.org/wiki/Standard_score . 
    Using np.mean() and np.std() with argument axis=0 can be useful for
    calculating the Z-score.
    - See also mandatory assignment 5 ("data processing")
    """
    pass


def convert_y_to_binary(y,y_value_true):
    """ Helper function to convert integer valued y to binary (0 or 1) valued vector 
    
    # Input arguments:
    y:              Integer valued NumPy vector, shape (n_samples,)
    y_value_true    Value of y which will be converted to 1 in output.
                    All other values are converted to 0.

    # Returns:
    y_binary:   Binary vector, shape (n_samples,)
                1 for values in y that are equal to y_value_true, 0 otherwise
    """
    return (y == y_value_true).astype(int)


def train_test_split(X,y,train_frac):
    """ Split dataset into training and testing datasets

    # Input arguments:
    X:              Dataset, shape (n_samples,n_features)   
    y:              Values to be predicted, shape (n_samples)
    train_frac:     Fraction of data to be used for training
    
    # Returns:
    (X_train,y_train):  Training dataset 
    (X_test,y_test):    Test dataset

    # Notes:
    - See also mandatory assignment 5 ("data processing")
    """
    pass


def accuracy(y_pred,y_true):
    """ Calculate accuracy of model based on predicted and true values 
    
    # Input arguments:
    y_pred:     Numpy array with predicted values, shape (n_samples,)
    y_true:     Numpy array with true values, shape (n_samples,)

    # Returns:
    accuracy:   Fraction of cases where the predicted values 
                are equal to the true values. Number in range [0,1]

    # Notes:
    See https://en.wikipedia.org/wiki/Sensitivity_and_specificity 
    """
    pass


###################
#   Perceptron
###################

class Perceptron:
    """ Perceptron model for classifying two linearly separable classes """

    def __init__(self):
        """ Initialize perceptron """
        pass

    def predict(self, x):
        """ Predict / calculate perceptron output for single observation / row 
        
        # Input arguments:
        x:      Numpy array, shape (n_features,)
                Corresponds to single row of data matrix X   
        
        # Returns:
        f:      Activation function output based on x: 
                I = (weights * x - bias)
                f(I) = 1 if I > 0,  0 otherwise
        """
        pass

    def train(self, X, y, learning_rate, max_epochs):
        """ Fit perceptron to training data X with binary labels y
        
        # Input arguments:
        X:              2D NumPy array, shape (n_samples, n_features)
        y:              NumPy vector, shape (n_samples), with 0 and 1 indicating which
                        class each sample in X corresponds to ("true" labels).
        learning_rate:  Learning rate, number in range (0.0 - 1.0)
        max_epochs:     Maximum number of epochs (integer, 1 or larger).

        # Notes:
        The algorithm should run until it converges on a stable set of weights
        (weights do not change from one iteration to the next). However, convergence
        is not guaranteed. The algorithm should only be allowed to run for a maximum 
        number of epochs (max_epochs) before terminating, to avoid infinite loops. 
        The attribute self.converged is used to indicate if the training converges
        (True/False). 
        """
        pass

    def print_info(self):
        """ Helper function for printing perceptron weights and convergence """
        print("Weights: " + str(self.weights))
        print("Bias: " + str(self.bias))
        print("Is converged: " + str(self.converged))

    def get_line_x_y(self):
        """ Helper function for calculating slope and intercept for decision boundary 
        
        # Returns:
        slope:      Slope of decision boundary line, assuming first feature as
                    x value and second feature as y value.  
        intercept:  Value where decision boundary line crosses y axis. 

        # Notes:
        - Only valid for perceptrons with 2 weights (based on X with 2 columns).
        - Solves w1*x + w2*y = b --> y = -(w1/w2) + (b/w2)
        """
        slope = -(self.weights[0]/self.weights[1])
        intercept = (self.bias / self.weights[1])

        return (slope,intercept)



##############################
#   Gini impurity functions
##############################

def gini_impurity(y):
    """ Calculate Gini impurity of a vector

    # Arguments:
    y   - 1D NumPy array with class labels

    # Returns:
    impurity  - Gini impurity, scalar in range [0,1)   
    
    # Notes:
    - Class labels can be encoded as integers (e.g. 0, 1, 2, ...) or 
    strings (e.g. "poor", "medium", "good")
    - Wikipedia ref.: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    pass   


def gini_impurity_reduction(y,left_mask):
    """ Calculate the reduction in mean impurity from a binary split
        
    # Arguments:
    y           - 1D numpy array
    left_mask   - 1D numpy boolean array, True for "left" elements, False for "right" 

    # Returns:
    impurity_reduction: Reduction in mean Gini impurity, scalar in range [0,0.5] 
                        Reduction is measured as _difference_ between Gini impurity for 
                        the original (not split) dataset, and the _weighted_ mean impurity 
                        for the two split datasets ("left" and "right").

    """
    pass


def best_split_feature_value(X,y):
    """ Find feature and value "split" that yields highest impurity reduction
    
    # Arguments:
    X:       NumPy feature matrix, shape (n_samples, n_features)
    y:       NumPy class label vector, shape (n_samples,)
    
    # Returns:
    impurity_reduction:     Reduction in Gini impurity for best split.
                            Zero if no split that reduces impurity exists. 
    feature_index:          Index of X column with best feature for split.
                            None if impurity_reduction = 0.
    feature_value:          Value of feature in X yielding best split of y
                            Dataset is split using X[:,feature_index] <= feature_value
                            None if impurity_reduction = 0.

    # Notes:
    - The method tests every possible feature and feature value found
    """
    pass


###################
#   Node classes 
###################

class DecisionTreeBranchNode(binarytree.Node):
    def __init__(self, feature_index, feature_value, left=None, right=None):
        """ Initialize decision node
        
        # Arguments:
        feature_index    Index of X column used in question
        feature_value    Value of feature used in question
        left             Node, root of left subtree
        right            Node, root of right subtree

        # Notes:
        - DecisionTreeBranchNode is a subclass of binarytree.Node. This
        has the advantage of inheriting useful methods for general binary
        trees, e.g. visualization through the __str__ method. 
        - Each decision node corresponds to a question of the form
        "is feature x <= value y". The features and values are stored as
        attributes "feature_index" and "feature_value". 
        - A string representation of the question is saved in the node's
        "value" attribute.
        """
        question_string = f'f{feature_index} <= {feature_value:.3g}'  # "General" format - fixed point/scientific
        super().__init__(value=question_string,left=left,right=right)
        self.feature_index = feature_index
        self.feature_value = feature_value

        
class DecisionTreeLeafNode(binarytree.Node):
    def __init__(self, y_value):
        """ Initialize leaf node

        # Arguments:
        y_value     class in dataset (e.g. integer or string) represented by leaf

        # Notes:
        The attribute "value" is set to the string representation of the value,
        to be used for visualization. The numeric value is stored in the attribute
        "y_value".
        """ 
        super().__init__(str(y_value))
        self.y_value = y_value


####################
#   Decision tree 
####################

class DecisionTree():
    def __init__(self):
        """ Initialize decision tree (no arguments)

        # Notes:
        - Attribute _y_dtype is used in DecisionTree to create empty arrays
        with same data type as the original y vector.
        """
        self._root = None
        self._y_dtype = None


    def __str__(self):
        """ Return string representation of decision tree (based on binarytree.Node.__str__())"""
        if self._root is not None:
            return(str(self._root))
        else:
            return '<Empty decision tree>'


    def train(self,X,y):
        """ Train decision tree based on labelled dataset 
        
        # Arguments:
        X        NumPy feature matrix, shape (n_samples, n_features)
        y        NumPy class label vector, shape (n_samples,)
        
        # Notes:
        Creates the decision tree by calling _build_tree() and setting
        the root node to the "top" DecisionTreeBranchNode. 
        
        """
        self._y_dtype = y.dtype
        self._root = self._build_tree(X,y)

    
    def _build_tree(self,X,y):
        """ Recursively build decision tree
        
        # Arguments:
        X        NumPy feature matrix, shape (n_samples, n_features)
        y        NumPy class label vector, shape (n_samples,)
        
        # Notes:
        - Determines the best possible binary split of the dataset. If no impurity
        reduction can be achieved, a leaf node is created, and its value is set to
        the most common class in y. If a split can achieve impurity reduction,
        a decision (branch) node is created, with left and right subtrees created by 
        recursively calling _build_tree on the left and right subsets.

        """
        # Find best binary split of dataset
        impurity_reduction, feature_index, feature_value = best_split_feature_value(X,y) 

        # If impurity can't be reduced further, create and return leaf node
        if impurity_reduction == 0:
            leaf_value = scipy.stats.mode(y,keepdims=False)[0] # Use most common class in dataset
            return DecisionTreeLeafNode(leaf_value)
        # If impurity _can_ be reduced, split dataset, build left and right 
        # branches, and return branch node.
        else:
            left_mask = X[:,feature_index] <= feature_value
            left = self._build_tree(X[left_mask],y[left_mask])
            right = self._build_tree(X[~left_mask],y[~left_mask])
            return DecisionTreeBranchNode(feature_index,feature_value,left,right)
    

    def predict(self,X):
        """ Predict class (y vector) for feature matrix X
        
        # Arguments:
        X        NumPy feature matrix, shape (n_samples, n_features)
        
        # Returns:
        y        NumPy class label vector (predicted), shape (n_samples,)
        
        # Notes:
        - "Entry level" method for prediction, used for first (recursive)
        call to _predict(), using root node as current node. 
        """
        return self._predict(X,self._root)
    

    def _predict(self,X,node):
        """ Predict class (y vector) for feature matrix X
        
        # Arguments:
        X       NumPy feature matrix, shape (n_samples, n_features)
        node    Node used to process the data. If the node is a leaf node,
                the data is classified with the value of the leaf node.
                If the node is a branch node, the data is split into left
                and right subsets, and classified by recursively calling
                _predict() on the left and right subsets.
        
        # Returns:
        y        NumPy class label vector (predicted), shape (n_samples,)

        # Notes:
        The prediction follows the following logic:

            if the node is a leaf node
                return y vector with all values equal to leaf node value
            else (the node is a branch node)
                split the dataset into left and right parts using node question
                predict classes for left and right datasets (using left and right branches)
                "stitch" predictions for left and right datasets into single y vector
                return y vector (length matching number of rows in X) 
        """
        pass
        
