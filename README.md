# CS4100_FinalProject

## Purpose:
The COVID-19 pandemic’s impact is difficult to predict due to volatile and inconsistent data accumulation. This can be attributed, in part, to fluctuating testing availability, changes in our understanding of symptoms, and varying administrative responses to the crisis around the world. Using machine learning, we can provide clear numerical predictions for crucial COVID-19 impact factors (mortality rate, expected case number, etc.) by running regressions on previously recorded statistics. Supervised machine learning looks at past data to make future conclusions. We can ac-complish this by setting a series of input variables which combine through some function f such that f([independent variables]) = D where D is some dependent (target) varia-ble.
The data values to be predicted are quantitative and non-binary, so regression is more applicable than clas-sification. The two regression algorithms chosen, Multivar-iable Linear Regression (MLR) and Neural Network Re-gression (NNR), are both well suited for this purpose. MLR uses a straightforward linear combination of the giv-en independent variables to calculate the target variable’s value but assumes linearity. The complexity of our input variables makes this linearity assumption unverifiable, therefore NNR is used to assess the quality of inputs and output in a non-linear way. By using both methods, the user can get a range of results that can be statistically use-ful in interpreting/predicting COVID-19 factors.

## Methodology:
We can concretely define the COVID-19 factor prediction problem by breaking it up into various components. First, we can apply the generic factor prediction problem to our COVID-19 dataset with two subproblems defined below. We also provide data descriptions for our input dataset and our output vector. 

**Problem 1**

Given a set of COVID statistics in an area for each time step t over the range t = 0 to t = n, predict the (number of cases | mortality rate) at time t = n+1.

**Problem 2**

Given a set of COVID statistics in an area for each time step t over the range t = 0 to t = n, calculate the impact of each COVID statistic with respect to the (number of cases | mortality rate).

**Inputs**

A list of states and their daily COVID attributes over a time interval. The statistics include the following:
- Date: The date in the form of MM/DD/YYYY that the following data corresponds to
- Province_State: Represents the name of the state/province
- Confirmed: Number of confirmed cases in the state/province
- Deaths: Number of deaths in the state/province
- Recovered: Number of recovered cases in the state/province
- Active: The active number of cases (Active cases = total cases - total recovered - total deaths)
- Incident_Rate: Cases per 100,000 persons
- Case_Fatality_Ratio (%): Number recorded deaths / Number cases.

The programmatic input is in the form of one csv file for each date. The rows correspond to each of the states/provinces. The data is obtained from the following link. Our input is a set of independent variables that would predict a dependent variable. We are designing the program to use any set of independent variables for any dependent variable. Therefore, any group of these variables can be used as the input.

**Outputs**

A set of feature-weight pairings that can be used to esti-mate the number of cases on a particular day given a set of feature values. f(W1*F1 , W2*F2 , … , Wn*Fn) = Vt


## File Structure:
`data_accumulation.py`: Contains a class that pulls and formats our dataset from Github.

`learning_agent.py`: Contains an Abstract Regression agent that implements common administrative processes (like pulling data) but leaves the regression algorithm unimplemented.

`homebrew_agent.py`: Contains a Multivariable Regression class that extends the Abstract Regression class and implements a MLR algorithm

`nn_agent.py`: Contains a Neural Network class that extends the Abstract Regression class and implements Neural Network Regression with a Sigmoid activator

`sklearn_agent.py`: An implementation of sklearn in our codebase's framework. Used to cross reference our output with a known Regression implementation.

`run.py`: main() that run's all 3 regressions and accumulates predictions for each day and for each state

`dataVisualization.py`: Code that generates graphs

`findweights.py`: main() for smaller scope analysis (single state over a date range or all states for a single day)

## Algorithm 1: Multivariable Linear Regression
Found in `homebrew_agent.py`

- `run_regression(X, y)`: uses the matrix of input vectors to train a weights vector in self.coefficients and a bias value in self.intercept
- `add_intercept(X)`: adds a intercept column to the input data X that will be used to store (learn) bias updates
- `predict(entry)`: called after training, takes in an array of independent variable values and returns a predicted target variable value using the below function:
 
***Calculate the 'C' coefficients and 'b' intercept of:***

f( C1(*variable1*), C2(*variable2*), ..., Cn(*variablen*), b ) = *regressand_variable*

Examples:

f( C1(Hospitalization Rate) ) = Mortality_Rate

f( C1(Tests_Given), C2(Previous_Case_Number) ) = Case_Number

Function:

`D = C1*P1 + C2*P2 + … + Cn*Pn + b`



## Algorithm 2: Neural Network regression
Found in `nn_agent.py`

- `run_regression(X, y)`: formats and scales the input matrices X and y and sends them to the training function
- `predict(entry)`: called after training, takes in an array of independent variable values and returns a predicted target variable value
- `sigmoid_activation(X)`: Formats the input data, calculates the dot product with the stored coefficients, and calculates the sigmoid
- `sigmoid_loss(X)`: calculates the 'derivative' of the sigmoid output -> used for updating the weight vector
- `train(X, y, num_cycles)`: uses the matrix of input vectors to train a weights vector in self.coefficients

***Activation Function:***

Sigmoid: `F(x) = 1 / (1 + e-x)`

Weight Updates: `P=(predicted_values-actual_values)*(predict-ed_values*(1-predicted_values))`
