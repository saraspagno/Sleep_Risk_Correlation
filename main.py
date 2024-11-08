import os

import pandas as pd
import constants
from database import DataBase
from graph import Graph
from risk import Risk
from sleep import Sleep


def merge_all_on_same_day(overall: dict, work_early: dict, woke_many_time: dict, sleep_latency: dict, risk: dict,
                          risk_appeal: dict):
    data = []
    for date in overall:
        if date in risk and date in risk_appeal:
            data.append(
                {'Day': date,
                 'Overall Sleep Score': overall[date],
                 'Woke Early Score': work_early[date],
                 'Woke Many Times Score': woke_many_time[date],
                 'Sleep Latency Score': sleep_latency[date],
                 'Risk Score': risk[date],
                 'Risk Appeal Score': risk_appeal[date]})
    return pd.DataFrame(data)


def get_all_correlations(group: str):
    """Turns a timestamp into a unique day without seconds.
    Args:
        group: the name of the directory (group) which contains the db files.
    Returns: correlations between sleep and binary risk, and correlations between sleep and continuous risk.
    """
    files = os.listdir(group)
    files.sort()
    all_merged_dfs = []
    for filename in files:
        file_path = os.path.join(group, filename)
        print(f"processing file: {file_path}")
        db = DataBase(file_path)

        # creating the Risk object, which will create a map between unique day and risk score
        risk = Risk(db)
        # creating the sleep object, which will create a map between unique day and sleep score
        sleep = Sleep(db)

        # merging the sleep and risk scores into a dictionary of sleep:risk, based on equal unique day
        merged_df = merge_all_on_same_day(sleep.overall, sleep.woke_early, sleep.woke_many_time, sleep.sleep_latency,
                                          risk.continuous_corr, risk.risk_appeal)
        all_merged_dfs.append(merged_df)

        # graph = Graph(merged_df, file_path)
        # graph.show_sleep_risk_regression()
        # graph.print_partial_corr()
    final_merged_df = pd.concat(all_merged_dfs, ignore_index=True)
    graph = Graph(final_merged_df, group)
    graph.show_regression("Woke Many Times Score")
    # graph.show_sleep_risk_regression()
    # graph.show_risk_risk_appeal_regression()
    return final_merged_df


def main():
    for group in constants.GROUPS:
        print(f"processing directory: {group}\n")
        get_all_correlations(group)


if __name__ == '__main__':
    main()
