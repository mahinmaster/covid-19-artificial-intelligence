import numpy as np
from numpy.linalg import LinAlgError

from WeightFinder.learning_agent import RegressionAgent


class HomebrewAgent(RegressionAgent):
    """
    Implements a multiple regression agent using a homebrewed regression algorithm.
    """

    def __init__(self, independent_variables, dependent_variable, data_acc=None):
        super().__init__(independent_variables, dependent_variable, data_acc)
        self.coefficients = []
        self.intercept = None
        self.agent = 'homebrew'

    def run_regression(self, X, y):
        """
        Runs a multiple regression over the independent variables in X to see their weighted relationship to the
        dependent variable in y.

        :param X: A List of colums (variables) whose weights will be found with respect to y
        :param y: A column (variable) that is dependent on the variables in X
        :returns: the list of independent-variable coefficients
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = self.add_intercept(X)
        try:
            weights = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        except LinAlgError:
            return self.coefficients
        self.intercept = weights[0]
        self.coefficients = weights[1:]
        return self.coefficients

    def add_intercept(self, X):
        """
        Add a column to the input that can be used to accumulate an intercept/bias value (target variable to independent variables)
        :param X: The independent variable input matrix
        :return: The input matrix with another column of all 1's
        """
        ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)

    def predict(self, entry):
        """
        Calculates the target variable value given a set of independent values (corresponding to the trained array)
        Formula: TargetValue = C1*entry[0] + C2*entry[1] +... + Cn*entry[n-1] + Y-intercept
        :param entry: a list of values that map to the crained coefficients [c1 ... cn]
        :return: a predicted value for the target variable
        """
        prediction = self.intercept
        for value, coeff in zip(entry, self.coefficients):
            prediction += (value * coeff)
        return prediction
