import os

import pandas as pd
import constants
from database import DataBase
from graph import Graph
from mood import Mood
from risk import Risk
from sleep import Sleep
import warnings


def merge_all_on_same_day(file_name: str, risk, sleep, mood):
    data = []
    # days in overall are also present in all other sleep scores.
    for date in sleep.overall:
        if date in risk.risk and date in risk.experience_value and date in mood.irritable:
            data.append(
                {'User': file_name,
                 'Day': date,
                 'Overall_Sleep_Score': sleep.overall[date],
                 'Woke_Early_Score': sleep.woke_early[date],
                 'Woke_Many_Times_Score': sleep.woke_many_time[date],
                 'Sleep_Latency_Score': sleep.sleep_latency[date],
                 'Risk_Score': risk.risk[date],
                 'Experience_Value': risk.experience_value[date],
                 'Valence': mood.valence[date],
                 'Arousal': mood.arousal[date],
                 'Anxious': mood.anxious[date],
                 'Elated': mood.elated[date],
                 'Sad': mood.sad[date],
                 'Irritable': mood.irritable[date],
                 'Energetic': mood.energetic[date]
                 })
    return pd.DataFrame(data)


def get_all_correlations():
    """Gets all correlations and shows the graphs.
    """

    all_groups_together = []
    dataframes = {}
    for group in constants.GROUPS:
        all_merged_dfs = []
        print(f"processing directory: {group}\n")
        files = os.listdir(group)
        files.sort()
        for filename in files:
            file_path = os.path.join(group, filename)
            db = DataBase(file_path)
            # creating the Risk object, which will create a map between unique day and risk score
            risk = Risk(db)

            # creating the sleep object, which will create a map between unique day and sleep score
            sleep = Sleep(db)

            # creating the sleep object, which will create a map between unique day and mood scores
            mood = Mood(db)

            # merging the sleep and risk scores into a dictionary of sleep:risk, based on equal unique day
            file_df = merge_all_on_same_day(file_path, risk, sleep, mood)
            all_merged_dfs.append(file_df)

        group_df = pd.concat(all_merged_dfs, ignore_index=True)
        dataframes[group] = group_df
        all_groups_together.append(group_df)

    all_groups_df = pd.concat(all_groups_together, ignore_index=True)
    dataframes["Both Groups"] = all_groups_df
    graph = Graph(all_groups_df, dataframes)
    graph.risk_experience_value_regression()
    graph.sleep_risk_regression()
    graph.correlation_between_sleep_mood()
    graph.mediation_sleep_mood_risk()
    graph.mediation_sleep_risk_mood()


def main():
    get_all_correlations()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
