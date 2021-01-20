import sys

from WeightFinder.homebrew_agent import HomebrewAgent
from WeightFinder.sklearn_agent import SklearnAgent
from WeightFinder.nn_agent import NeuralNetworkAgent

USAGE = "Usage: ./findweights -agent <sklearn | homebrew> -state [U.S. state] -d1 <Initial date> -d2 [Ending date]"


def parse_args(argv):
    """
    Parses command line arguments into interpretable values. Prints usage message if the given args. aren't valid

    :param argv: the command line arguments
    :return: strings representing the type of agent, the state to read, and the start/end days of the time interval.
             Note: some of these args are optional and will be None.
    """
    agent_type = None
    state = None
    start_day = None
    end_day = None
    try:
        for idx, arg in enumerate(argv):
            if arg == "-agent":
                agent_type = argv[idx + 1]
            elif arg == "-state":
                state = argv[idx + 1]
            elif arg == "-d1":
                start_day = argv[idx + 1]
            elif arg == "-d2":
                end_day = argv[idx + 1]
    except Exception:
        print(USAGE)
        sys.exit(1)
    if (agent_type and start_day and not state and not end_day) or (agent_type and start_day and state and end_day):
        return agent_type, state, start_day, end_day
    else:
        print(USAGE)
        sys.exit(1)


def run_agent(agent, state, start_day, end_day):
    """
    Depending on the given arguments, either calculates aggregated variable weights across all 50 states
    on a given day or calculates variable weights for a specific state over a given time interval
        usage 1: ./findweights -agent <sklearn | homebrew> -d1 <MM-DD-YYYY>
        usage 2:./findweights -agent <sklearn | homebrew> -state [U.S. state] -d1 <MM-DD-YYYY> -d2 [MM-DD-YYYY]

    :param agent: an instance of some regression agent that will be used to calcualate variable weights
    :param state: an optional value containing the state to look at
    :param start_day: The first day on the time interval to run a regression
    :param end_day: The last day on the time interval to run a regression
    :return: a dictionary mapping the independent variables with the weighted variable coefficients

    """
    if state and start_day and end_day:
        coefficients = agent.run_for_us_state(state, start_day, end_day)
    elif start_day and not state and not end_day:
        coefficients = agent.run_for_day('04-12-2020')
    else:
        print(USAGE)
        sys.exit(1)
    return coefficients


def main(argv):
    """
    Runs a multiple regression using the specified values in argv.

    :param argv: the command line arguments
    :return: a dictionary mapping the independent variables with the weighted variable coefficients
    """
    agent_type, state, start_day, end_day = parse_args(argv)
    independent_variables = ['Confirmed', 'Hospitalization_Rate']
    #independent_variables = ['Confirmed']
    dependent_variable = 'Mortality_Rate'
    if agent_type == "sklearn":
        agent = SklearnAgent(independent_variables, dependent_variable)
        coefficients = run_agent(agent, state, start_day, end_day)
    elif agent_type == "homebrew":
        agent = HomebrewAgent(independent_variables, dependent_variable)
        coefficients = run_agent(agent, state, start_day, end_day)
    elif agent_type == "neural":
        agent = NeuralNetworkAgent(independent_variables, dependent_variable)
        coefficients = run_agent(agent, state, start_day, end_day)
    print(coefficients)
    print(agent.predict([3667, 12.26494527]))

    return coefficients


if __name__ == "__main__":
    main(sys.argv[1:])
