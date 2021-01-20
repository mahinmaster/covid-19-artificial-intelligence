import sys
import pandas as pd
from WeightFinder.homebrew_agent import HomebrewAgent
from WeightFinder.sklearn_agent import SklearnAgent
from WeightFinder.nn_agent import NeuralNetworkAgent
from sklearn.metrics import r2_score
import openpyxl


USAGE = "Usage: ./findweights -agent <sklearn | homebrew> -state [U.S. state] -d1 <Initial date> -d2 [Ending date]"

init_state, start_day, end_day = 'Alabama' , '04-12-2020' , '12-14-2020'
independent_variables = ['Confirmed', 'Hospitalization_Rate']
dependent_variable = 'Mortality_Rate'
run_type = True
helper_agent = HomebrewAgent(independent_variables,dependent_variable)
helper_agent.pull_data(start_day, end_day)
dataAcc = helper_agent.data_acc
output_path = "/Users/andrewduffy/Documents/Fall2020NEU/CS4100/Final Project/CS4100_FinalProject/regression_output.xlsx"


def run_agent(agent, state):
    """
    Depending on the given arguments, either calculates aggregated variable weights across all 50 states
    on a given day or calculates variable weights for a specific state over a given time interval

    :param agent: an instance of some regression agent that will be used to calcualate variable weights
    :param state: an optional value containing the state to look at
    :return: a dictionary mapping the independent variables with the weighted variable coefficients

    """
    if run_type:
        coefficients = agent.run_for_us_state(state, start_day, end_day)

    else:
        coefficients = agent.run_for_day(start_day)

    return coefficients

def run_every_state():
    full_table = pd.DataFrame(columns=['agent', 'state', 'predicted', 'actual', 'error'])
    for state in dataAcc.data['04-12-2020']['Province_State']:
        print(state)
        sk_agent = SklearnAgent(independent_variables, dependent_variable,dataAcc)
        sk_coefficients = run_agent(sk_agent, state)

        hb_agent = HomebrewAgent(independent_variables, dependent_variable,dataAcc)
        hb_coefficients = run_agent(hb_agent, state)

        nn_agent = NeuralNetworkAgent(independent_variables, dependent_variable,dataAcc)
        nn_coefficients = run_agent(nn_agent, state)

        if sk_coefficients and hb_coefficients and nn_coefficients:
            try:
                print("Sklearn:")
                sk_prediction = get_accuracy_for_state(sk_agent, state)
                #sk_prediction = sk_agent.predict([3667, 12.26494527])
                print(sk_prediction)
                full_table = full_table.append(sk_prediction)

                print("Homebrew:")
                hb_prediction = get_accuracy_for_state(hb_agent, state)
                #hb_prediction = hb_agent.predict([3667, 12.26494527])
                print(hb_prediction)
                full_table = full_table.append(hb_prediction)

                print("NeuralNetwork:")
                nn_prediction = get_accuracy_for_state(nn_agent, state)
                #nn_prediction = nn_agent.predict([3667, 12.26494527])
                print(nn_prediction)
                full_table = full_table.append(nn_prediction)

            except:
                print("error with:",state)
                continue
    full_table.to_excel(output_path,sheet_name='Accumulated')

        #sys.exit(1)

def get_accuracy_for_state(agent,state):
    state_data = dataAcc.get_state(state)
    state_data = state_data.dropna()
    accuracy_table = pd.DataFrame(columns=['agent','state','predicted','actual'])
    for index,row in state_data.iterrows():
        format_row = row[independent_variables].to_numpy()
        prediction = agent.predict(format_row)
        actual = row[dependent_variable]
        accuracy_table = accuracy_table.append({'agent':agent.agent,'state':state,'predicted':prediction,'actual':actual},ignore_index=True)
    accuracy_table['error'] = calculate_RMSE(accuracy_table['predicted'],accuracy_table['actual'])
    return accuracy_table

def calculate_RMSE(predicted,actual):
    return r2_score(y_true=actual,y_pred=predicted,multioutput='variance_weighted')


def main():
    """
    Runs a regression and accumulates data for all states
    """
    run_every_state()
    return


if __name__ == "__main__":
    main()
