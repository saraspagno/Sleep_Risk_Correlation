import os

from scipy.stats import linregress

import constants
from accuracy import Accuracy
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
    sleep_w_risk_binary_all, sleep_w_risk_continuous_all, day_accuracy_map = {}, {}, {}
    files = os.listdir(group)
    files.sort()
    for filename in files:
        file_path = os.path.join(group, filename)
        print(f"processing file: {file_path}")
        db = DataBase(file_path)

        # creating Accuracy object, which will create a map of accuracies over time
        accuracy = Accuracy()

        # creating the Risk object, which will create a map between unique day and risk score
        risk = Risk(db, accuracy)
        # creating the sleep object, which will create a map between unique day and sleep score
        sleep = Sleep(db)

        # merging the sleep and risk scores into a dictionary of sleep:risk, based on equal unique day
        merged_risk_continuous = merge_sleep_risk_on_date(sleep.correlations, risk.continuous_corr)
        merged_risk_binary = merge_sleep_risk_on_date(sleep.correlations, risk.binary_corr)
        graph = Graph(merged_risk_binary, merged_risk_continuous, accuracy.accuracy_corr, file_path)
        graph.show_sleep_risk_regression()
        graph.show_accuracy_time_regression()
        if sleep_and_risk_correlated(merged_risk_continuous):
            graph.show_accuracy_time_regression()


def check_accuracy(accuracy_map: dict, threshold: float = 90):
    total_accuracy = sum(accuracy_map.values()) / len(accuracy_map)
    return total_accuracy >= threshold


def accuracy_is_improving(accuracy_map: dict) -> bool:
    time = list(accuracy_map.keys())
    accuracies = list(accuracy_map.values())
    slope, _, _, p_value, _ = linregress(time, accuracies)
    return slope > 0 and p_value <= 0.05


def sleep_and_risk_correlated(sleep_risk: dict) -> bool:
    sleep = list(sleep_risk.keys())
    risk = list(sleep_risk.values())
    slope, _, _, p_value, _ = linregress(sleep, risk)
    return slope < 0 and p_value <= 0.05


def main():
    for group in constants.GROUPS:
        print(f"processing directory: {group}\n")
        get_all_correlations(group)


if __name__ == '__main__':
    main()
