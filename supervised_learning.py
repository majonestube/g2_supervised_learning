import csv
import random as rnd
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats  # Used for "mode" - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
from decision_tree_nodes import DecisionTreeBranchNode, DecisionTreeLeafNode
from matplotlib import lines
from numpy.typing import NDArray

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


def read_data(file_path) -> tuple[NDArray, NDArray]:
    """Read data from CSV file, remove rows with missing data, and normalize

    Returns
    -------
    X: NDArray
        Numpy array, shape (n_samples,4), where n_samples is number of rows
        in the dataset. Contains the four numeric columns in the dataset
        (bill length, bill depth, flipper length, body mass).
        Each column (each feature) is normalized by subtracting the column mean
        and dividing by the column std.dev. ("z-score").
        Rows with missing data ("NA") are discarded.
    y: NDarray
        Numpy array, shape (n_samples,)
        Contains integer values (0, 1 or 2) representing the penguin species

    Notes
    -----
    Z-score normalization: https://en.wikipedia.org/wiki/Standard_score .
    """
    with open(file_path, "r", newline="") as file:
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
    header = data[0]
    data = np.array(data[1:])

    is_col = np.isin(header, ["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
    data_filt = data[:, is_col]
    is_na = np.any(data_filt == "NA", axis=1)  # True if any element in row is NA
    data_filt = data_filt[~is_na, :]

    X = data_filt[:, 1:].astype(float)

    y_text = data_filt[:, 0]  #
    y_unique = list(np.unique(y_text))
    y = np.array([y_unique.index(species) for species in y_text]) # Convert text to numbers

    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X_norm, y


def convert_y_to_binary(y: NDArray, y_value_true: int) -> NDArray:
    """Convert integer valued y to binary (0 or 1) valued vector

    Parameters
    ----------
    y: NDArray
        Integer valued NumPy vector, shape (n_samples,)
    y_value_true: int
        Value of y which will be converted to 1 in output.
        All other values are converted to 0.

    Returns
    -------
    y_binary: NDArray
        Binary vector, shape (n_samples,)
        1 for values in y that are equal to y_value_true, 0 otherwise
    """

    y_temp = np.where(y == y_value_true, 1, 0)
    y_binary = y_temp.astype(int)
    
    return y_binary


def train_test_split(
    X: NDArray, y: NDArray, train_frac: float
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """Shuffle and split dataset into training and testing datasets

    Parameters
    ----------
    X: NDArray
        Dataset, shape (n_samples,n_features)
    y: NDArray
        Values to be predicted, shape (n_samples)
    train_frac: float
        Fraction of data to be used for training

    Returns
    -------
    (X_train,y_train): tuple[NDArray, NDArray]]
        Training dataset
    (X_test,y_test): tuple[NDArray, NDArray]]
        Test dataset
    """
    pass


def accuracy(y_pred: NDArray, y_true: NDArray) -> float:
    """Calculate accuracy of model based on predicted and true values

    Parameters
    ----------
    y_pred: NDArray
        Numpy array with predicted values, shape (n_samples,)
    y_true: NDArray
        Numpy array with true values, shape (n_samples,)

    Returns
    -------
    accuracy: float
        Fraction of cases where the predicted values
        are equal to the true values. Number in range [0,1]

    # Notes:
    See https://en.wikipedia.org/wiki/Accuracy_and_precision#In_classification
    """
    pass


##############################
#   Gini impurity functions
##############################


def gini_impurity(y: NDArray) -> float:
    """Calculate Gini impurity of a vector

    Parameters
    ----------
    y: NDArray, integers
        1D NumPy array with class labels

    Returns
    -------
    impurity: float
        Gini impurity, scalar in range [0,1)

    # Notes:
    - Wikipedia ref.: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """

    n = y.shape[0]
    impurity = 1
    for x in np.unique(y):
        y_binary = convert_y_to_binary(y, x)
        n_x = np.count_nonzero(y_binary == 1)
        impurity -= (n_x/n)**2
    return impurity




def gini_impurity_reduction(y: NDArray, left_mask: NDArray) -> float:
    """Calculate the reduction in mean impurity from a binary split

    Parameters
    ----------
    y: NDArray
        1D numpy array
    left_mask: NDArray
        1D numpy boolean array, True for "left" elements, False for "right"

    Returns
    -------
    impurity_reduction: float
        Reduction in mean Gini impurity, scalar in range [0,0.5]
        Reduction is measured as _difference_ between Gini impurity for
        the original (not split) dataset, and the _weighted mean impurity_
        for the two split datasets ("left" and "right").

    """
    original_impurity = gini_impurity(y)
    y_left = y[left_mask]
    y_right = y[~left_mask]
    gi_left = gini_impurity(y_left)
    gi_right = gini_impurity(y_right)
    n_left = y_left.shape[0]
    n_right = y_right.shape[0]
    n = y.shape[0]

    return (original_impurity - ((n_left * gi_left) + (n_right * gi_right))/n)

def best_split_feature_value(X: NDArray, y: NDArray) -> tuple[float, int, float]:
    """Find feature and value "split" that yields highest impurity reduction

    Parameters
    ----------
    X: NDArray
        NumPy feature matrix, shape (n_samples, n_features)
    y: NDArray
        NumPy class label vector, shape (n_samples,)

    Returns
    -------
    impurity_reduction: float
        Reduction in Gini impurity for best split.
        Zero if no split that reduces impurity exists.
    feature_index: int
        Index of X column with best feature for split.
        None if impurity_reduction = 0.
    feature_value: float
        Value of feature in X yielding best split of y
        Dataset is split using X[:,feature_index] <= feature_value
        None if impurity_reduction = 0.

    Notes
    -----
    The method checks every possible combination of feature and
    existing unique feature values in the dataset.
    """
    feature_index = None
    feature_value = None
    impurity_reduction = -(float('inf'))
    for j in range(X.shape[1]):
        for value in np.unique(X[:, j]):
            left_mask = (X[:,j] <= value)
            gi_reduction = gini_impurity_reduction(y, left_mask)
            if gi_reduction > impurity_reduction:
                impurity_reduction = gi_reduction
                feature_index = j
                feature_value = value
    return impurity_reduction, feature_index, feature_value

###################
#   Perceptron
###################


class Perceptron:
    """Perceptron model for classifying two classes

    Attributes
    ----------
    weights: NDArray
        Array, shape (n_features,), with perceptron weights
    bias: float
        Perceptron bias value
    converged: bool | None
        Boolean indicating if Perceptron has converged during training.
        Set to None if Perceptron has not yet been trained.
    """

    def __init__(self):
        """Initialize perceptron"""
        pass

    def predict_single(self, x: NDArray) -> int:
        """Predict / calculate perceptron output for single observation / row x
        <Write rest of docstring here>
        """
        pass

    def predict(self, X: NDArray) -> NDArray:
        """Predict / calculate perceptron output for data matrix X
        <Write rest of docstring here>
        """
        pass

    def train(self, X: NDArray, y: NDArray, learning_rate: float, max_epochs: int):
        """Fit perceptron to training data X with binary labels y
        <Write rest of docstring here>
        """
        pass

    def decision_boundary_slope_intercept(self) -> tuple[float, float]:
        """Calculate slope and intercept for decision boundary line (2-feature data only)
        <Write rest of docstring here>
        """
        pass


####################
#   Decision tree
####################


class DecisionTree:
    """Decision tree model for classification

    Attributes
    ----------
    _root: DecisionTreeBranchNode | None
        Root node in decision tree
    """

    def __init__(self):
        """Initialize decision tree"""
        self._root = None

    def __str__(self) -> str:
        """Return string representation of decision tree (based on binarytree.Node.__str__())"""
        if self._root is not None:
            return str(self._root)
        else:
            return "<Empty decision tree>"

    def fit(self, X: NDArray, y: NDArray):
        """Train decision tree based on labelled dataset

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray, integers
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        Creates the decision tree by calling _build_tree() and setting
        the root node to the "top" DecisionTreeBranchNode.

        """
        self._root = self._build_tree(X, y)

    def _build_tree(self, X: NDArray, y: NDArray):
        """Recursively build decision tree

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        - Determines the best possible binary split of the dataset. If no impurity
        reduction can be achieved, a leaf node is created, and its value is set to
        the most common class in y. If a split can achieve impurity reduction,
        a decision (branch) node is created, with left and right subtrees created by
        recursively calling _build_tree on the left and right subsets.

        """
        # Find best binary split of dataset
        impurity_reduction, feature_index, feature_value = best_split_feature_value(
            X, y
        )

        # If impurity can't be reduced further, create and return leaf node
        if impurity_reduction == 0:
            leaf_value = scipy.stats.mode(y, keepdims=False)[0]
            return DecisionTreeLeafNode(leaf_value)

        # If impurity _can_ be reduced, split dataset, build left and right
        # branches, and return branch node.
        else:
            left_mask = X[:, feature_index] <= feature_value
            left = self._build_tree(X[left_mask], y[left_mask])
            right = self._build_tree(X[~left_mask], y[~left_mask])
            return DecisionTreeBranchNode(feature_index, feature_value, left, right)

    def predict(self, X: NDArray):
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray, integers
            NumPy class label vector (predicted), shape (n_samples,)
        """
        if self._root is not None:
            return self._predict(X, self._root)
        else:
            raise ValueError("Decision tree root is None (not set)")

    def _predict(
        self, X: NDArray, node: Union["DecisionTreeBranchNode", "DecisionTreeLeafNode"]
    ) -> NDArray:
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        node: "DecisionTreeBranchNode" or "DecisionTreeLeafNode"
            Node used to process the data. If the node is a leaf node,
            the data is classified with the value of the leaf node.
            If the node is a branch node, the data is split into left
            and right subsets, and classified by recursively calling
            _predict() on the left and right subsets.

        Returns
        -------
        y: NDArray
            NumPy class label vector (predicted), shape (n_samples,)

        Notes
        -----
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


############
#   MAIN
############

if __name__ == "__main__":
    # Demonstrate your code / solutions here.
    # Be tidy; don't cut-and-paste lots of lines.
    # Experiments can be implemented as separate functions that are called here.

    X, y = read_data('palmer_penguins.csv')
    
