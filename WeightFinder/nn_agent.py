import numpy as np
from sklearn.preprocessing import MinMaxScaler

from WeightFinder.learning_agent import RegressionAgent


class NeuralNetworkAgent(RegressionAgent):
    """
    A neural network implementation using one hidden layer and sigmoid activation
    """

    def __init__(self, independent_variables, dependent_variable,data_acc = None):
        """
        Initialize parameters and weight vectors
        NOTE: all 'scalars' are used for data normalization
        :param independent_variables: The list of independent variables
        :param dependent_variable: The list of dependent variables
        """
        super().__init__(independent_variables, dependent_variable,data_acc)
        self.agent = 'neural'
        # Accumulate scalars for the input matrices (used to map large integers to a 0-1 numberline)
        self.input_scalars = []
        for i in independent_variables:
            self.input_scalars.append(MinMaxScaler(feature_range=(0, 1)))

        # Accumulate a scalar for the output matrix
        self.output_scalar = MinMaxScaler(feature_range=(0, 1))

        self.coefficients = np.random.random((len(independent_variables), 1))

    def sigmoid_activation(self, X):
        """
        Formats the input data, calculates the dot product with the stored coefficients, and calculates the sigmoid
        :param X: Input data
        :return: A sigmoid value 0 < n < 1
        """
        X = np.dot(X.astype(np.float128), self.coefficients)
        return 1 / (1 + np.exp(-X))

    def sigmoid_loss(self, X):
        """
        Calculates the 'derivative' of the sigmoid output -> used for updating the weight vector
        :param X: vector to be changed
        :return: the derivative of the sigmoid output
        """
        return X * (1 - X)

    def train(self, X, y, num_cycles):
        """
        Trains the NN's weight vector (self.coefficients) over the given input data
        :param X: Input data for Independent Variables
        :param y: Actual output data corresponding to the independent variables
        :param num_cycles: number if iterations for training
        """
        for _ in range(num_cycles):
            prediction = self.sigmoid_activation(X)
            predicted_vs_actual = y - prediction
            diff = np.dot(X.T, predicted_vs_actual * self.sigmoid_loss(prediction))
            self.coefficients += diff

        return prediction

    def run_regression(self, X, y):
        """
        Runs a Neural Network regression over the independent variables in X to see their weighted relationship to the
        dependent variable in y.
        :param X: A matrix with each column being an indepentent variable
        :param y: A 1d matrix of target variable values
        :return: a list of calculated weights for the dependent variables
        """
        X = X.to_numpy(dtype=np.float128)
        y = np.array([y], dtype=np.float128).T

        for i in range(X.shape[1]):
            self.input_scalars[i].fit(X[:, i].reshape(-1, 1))
            X[:, i] = self.input_scalars[i].transform(X[:, i].reshape(-1, 1))[:, 0]
        self.output_scalar.fit(y)

        y = self.output_scalar.transform(y)
        self.train(X, y, 10000)
        self.variable_weights = self.coefficients

        return self.variable_weights

    def predict(self, entry: list):
        """
        Calculates the target variable value given a set of independent values (corresponding to the trained array)
        Formula: TargetValue = C1*entry[0] + C2*entry[1] +... + Cn*entry[n-1] + Y-intercept
        :param entry: a list of values that map to the crained coefficients [c1 ... cn]
        :return: a predicted value for the target variable
        """
        prediction = 0

        weighted_entries = []
        for param, scalar in zip(entry, self.input_scalars):
            weighted_entries.append(scalar.transform(np.array([param]).reshape(-1, 1))[0])

        for value, coeff in zip(weighted_entries, self.coefficients):
            prediction += (value * coeff)

        unscaled = self.output_scalar.inverse_transform([prediction])
        return unscaled[0][0]
