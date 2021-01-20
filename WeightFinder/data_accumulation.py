from typing import Dict, List

import pandas as pd


# Note: update macOS ssl certificates or you will get an error during the read_csv function
# /Applications/Python\ 3.6/Install\ Certificates.command

class DataAcc:
    """
    This class is a "Data Accumulator" for COVID-19 statistics from the Johns Hopkins CSSE github. The github folder
    linked in self.root_url accumulates COVID-19 statistics for each U.S. state from around the web daily and
    stores them in .csv files.

    Order of operations:
        __init__()
        pull_data()
        get_day() or get_state()
    """
    def __init__(self):
        self.root_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data" \
                        "/csse_covid_19_daily_reports_us"
        self.data: Dict[str, List] = {}

    def pull_data(self, start_day: str, end_day: str, fields: List[str]):
        """
        Reads all .csv data from the JH CSSE github folder on the time interval start_day to end_day. Puts each day's
        data into a pandas DataFrame, which is then indexed by day in self.data. Also filters down each day's data to
        the columns specified in fields.

        :param start_day: the inclusive first day of the timeframe to pull daily data from. Format: 'MM-DD-YYYY'
        :param end_day: the inclusive last day of the timeframe to pull daily data from. Format: 'MM-DD-YYYY'
        :param fields: the list of fields (i.e. column names) to pull from each .csv file
        :return: a tuple in the format (number of days whose data was pulled, list of days that had data). This return
                 is mainly for testing because this function also updates self.data, which is used for calculations
        """
        date_range = pd.date_range(start=start_day, end=end_day)
        for date in date_range:
            f_date = date.strftime("%m-%d-%Y")
            try:
                url = self.root_url + "/" + f_date + ".csv"
                self.data[f_date] = pd.read_csv(url, error_bad_lines=False)[fields]
            except Exception as e:
                print(e)
                print("Error loading:", f_date)
        # print(len(self.data))

        return (len(self.data), list(self.data.keys()))

    def get_day(self, day):
        """
        Returns the specified day's corresponding DataFrame containing its COVID statistics.

        :param day: the day to pull stats from
        :return: A Pandas DataFrame with each state's stats from the given day
        """
        try:
            return self.data[day]
        except Exception:
            raise ValueError("Couldn't find the provided date in the dataset.")

    def get_state(self, state):
        """
        Returns a DataFrame with the input state's accumulated covid stats for every day.

        :param state: the state to accumulate data for
        :return: A pandas DataFrame containing the accumulated daily covid data for the given state
        """
        state_acc = pd.DataFrame()
        for date, table in self.data.items():
            try:
                row = table.loc[table['Province_State'] == state]
                state_acc = state_acc.append(row)
            except Exception:
                print(state, "not found on", date)
                continue
        return state_acc
