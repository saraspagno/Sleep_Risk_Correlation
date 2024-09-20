import os
import constants
from database import DataBase
from graph import Graph
from risk import Risk
from sleep import Sleep


def merge_sleep_risk_on_date(sleep_values: dict, risk_values: dict):
    """Receives sleep and risk dictionaries, mapping unique days to sleep and risk scores respectively,
    and returns a merged dictionary of sleep with its own risk score (based on equal unique day).
    Args:
        sleep_values: the map of unique day and sleep score.
        risk_values: the map of unique day and risk score.
    Returns: a map mapping between sleep and risk on equal day.
    """
    score_map = {}
    for date in sleep_values:
        if date in risk_values:
            score_map[sleep_values[date]] = risk_values[date]
    return score_map


def get_all_correlations(group: str):
    """Turns a timestamp into a unique day without seconds.
    Args:
        group: the name of the directory (group) which contains the db files.
    Returns: correlations between sleep and binary risk, and correlations between sleep and continuous risk.
    """
    sleep_w_risk_binary_all, sleep_w_risk_continuous_all = {}, {}
    files = os.listdir(group)
    files.sort()
    for filename in files:
        file_path = os.path.join(group, filename)
        print(f"processing file: {file_path}")
        db = DataBase(file_path)
        # creating the Risk object, which will create a map between unique day and risk score
        risk = Risk(db)
        # creating the sleep object, which will create a map between unique day and sleep score
        sleep = Sleep(db)

        # merging the sleep and risk scores into a dictionary of sleep:risk, based on equal unique day
        sleep_w_risk_binary_all.update(merge_sleep_risk_on_date(sleep.correlations, risk.binary_corr))
        sleep_w_risk_continuous_all.update(merge_sleep_risk_on_date(sleep.correlations, risk.continuous_corr))
    return sleep_w_risk_binary_all, sleep_w_risk_continuous_all


def main():
    for group in constants.GROUPS:
        print(f"processing directory: {group}\n")
        binary, continuous = get_all_correlations(group)
        graph = Graph(binary, continuous, group)
        graph.show_regression()
        graph.show_box_plot()


if __name__ == '__main__':
    main()
