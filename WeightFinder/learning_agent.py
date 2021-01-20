from abc import ABC, abstractmethod

from WeightFinder.data_accumulation import DataAcc


class RegressionAgent(ABC):
    """
    A generic regression agent that is missing a regression algorithm. Extend this class and implement run_regression
    to calculate variable weights and make predictions.
    """
    def __init__(self, independent_variables, dependent_variable,data_acc = None):
        self.X = independent_variables
        self.y = dependent_variable
        if not data_acc:
            self.data_acc = DataAcc()
        else:
            self.data_acc = data_acc
        self.variable_weights = []

    def run_for_us_state(self, state, start_day, end_day):
        """
        Calculates the variable weights for the given state over a daily time-interval

        :param start_day: The first day of the timeseries to run a regression on
        :param end_day: the last day of the timeseries to run a regression on
        :param state: a US state name. ex: 'Massachusetts', 'Kansas', 'California', ...
        :return: a dictionary containing the regressed coefficients mapped to the variable name
        """
        if len(self.data_acc.data) == 0:
            self.pull_data(start_day, end_day)
        data = self.data_acc.get_state(state)
        data = data.dropna()
        X = data[self.X]
        y = data[self.y]
        if len(X) < 1:
            return None
            #raise ValueError(f"No data found for {state} over the date range {start_day} to {end_day}.")
        self.variable_weights = self.run_regression(X, y)
        if len(self.variable_weights) > 0:
            variable_weights = {self.X[i]: self.variable_weights[i] for i in range(len(self.X))}
            return variable_weights
        else:
            return None

    def run_for_day(self, date):
        """
        Runs a multiple-variable regression across all 50 states on a particular day and finds the variable coefficients

        :param date: the day whose variables will be used to run the regression
        :return: a dictionary containing the regressed coefficients mapped to the variable name
        """
        self.pull_data(date, date)
        data = self.data_acc.get_day(date)
        data = data.dropna()
        X = data[self.X]
        y = data[self.y]
        if len(X) < 1:
            raise ValueError(f"No data found for {date}.")
        self.variable_weights = self.run_regression(X, y)
        out = {self.X[i]: self.variable_weights[i] for i in range(len(self.X))}
        return out

    def pull_data(self, start_day, end_day):
        """
        Pulls the data from github and filters it to the columns needed for this regression

        :param start_day: The first day on the time interval to pull data for
        :param end_day: The last day on the time interval to pull data for
        """
        fields = ['Province_State'] + self.X + [self.y]
        self.data_acc.pull_data(start_day, end_day, fields)

    @abstractmethod
    def run_regression(self,X,y):
        """
        Runs a multiple regression over the independent variables in X to see their weighted relationship to the
        dependent variable in y.

        :param X: A List of colums (variables) whose weights will be found with respect to y
        :param y: A column (variable) that is dependent on the variables in X
        :raises NotImplementedError: This class is abstract. Call one of its extensions.
        """
        raise NotImplementedError("Do not use this super class to run a regression. Instead, call homebrew_agent or sklearn_agent")
