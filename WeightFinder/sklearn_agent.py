from WeightFinder.learning_agent import RegressionAgent
from sklearn import linear_model


class SklearnAgent(RegressionAgent):
    """
    Implements a multiple regression agent using the sklearn library's regression functions.
    """
    def __init__(self, independent_variables, dependent_variable,data_acc = None):
        super().__init__(independent_variables, dependent_variable,data_acc)
        self.linear_regression = None
        self.agent = 'sklearn'

    def run_regression(self, X, y):
        """
        Runs a multiple regression over the independent variables in X to see their weighted relationship to the
        dependent variable in y.

        :param X: A List of colums (variables) whose weights will be found with respect to y
        :param y: A column (variable) that is dependent on the variables in X
        :return: The list of coefficients calculated by the regression algorithm
        """
        self.linear_regression = linear_model.LinearRegression()
        self.linear_regression.fit(X, y)
        return self.linear_regression.coef_

    def predict(self,entry:list):
        """

        :param entry:
        :return:
        """
        return self.linear_regression.predict([entry])[0]



