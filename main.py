import os

import pandas as pd
import constants
from database import DataBase
from graph import Graph
from risk import Risk
from sleep import Sleep


def merge_all_on_same_day(overall: dict, work_early: dict, woke_many_times: dict, sleep_latency: dict, risk: dict,
                          risk_appeal: dict):
    """Merges multiple dictionaries in one dataframe.
    Args:
      overall: the overall sleep score.
      work_early: the woke early sleep score.
      woke_many_times: the woke many times sleep score.
      sleep_latency: the sleep latency score.
      risk: the risk score.
      risk_appeal: the risk appeal score.
    """
    data = []
    # days in overall are also present in all other sleep scores.
    for date in overall:
        if date in risk and date in risk_appeal:
            data.append(
                {'Day': date,
                 'Overall Sleep Score': overall[date],
                 'Woke Early Score': work_early[date],
                 'Woke Many Times Score': woke_many_times[date],
                 'Sleep Latency Score': sleep_latency[date],
                 'Risk Score': risk[date],
                 'Risk Appeal Score': risk_appeal[date]})
    return pd.DataFrame(data)


def get_all_correlations():
    """Gets all correlations and shows the graphs.
    """

    for group in constants.GROUPS:
        all_merged_dfs = []
        print(f"processing directory: {group}\n")
        files = os.listdir(group)
        files.sort()
        for filename in files:
            file_path = os.path.join(group, filename)
            # print(f"processing file: {file_path}")
            db = DataBase(file_path)
            # creating the Risk object, which will create a map between unique day and risk score
            risk = Risk(db)
            # creating the sleep object, which will create a map between unique day and sleep score
            sleep = Sleep(db)

            # merging the sleep and risk scores into a dictionary of sleep:risk, based on equal unique day
            merged_df = merge_all_on_same_day(sleep.overall, sleep.woke_early, sleep.woke_many_time,
                                              sleep.sleep_latency,
                                              risk.risk, risk.risk_appeal)
            all_merged_dfs.append(merged_df)
        final_merged_df = pd.concat(all_merged_dfs, ignore_index=True)
        graph = Graph(final_merged_df, group)
        graph.show_risk_risk_appeal_regression()
        graph.partial_correlation()
        graph.pair_plot()

def main():
    get_all_correlations()


if __name__ == '__main__':
    main()
