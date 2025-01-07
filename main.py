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
        if date in risk.risk and date in risk.risk_appeal and date in mood.valence and date in mood.arousal and date in mood.anxious and date in mood.valence:
            data.append(
                {'User': file_name,
                 'Day': date,
                 'Overall_Sleep_Score': sleep.overall[date],
                 'Woke_Early_Score': sleep.woke_early[date],
                 'Woke_Many_Times_Score': sleep.woke_many_time[date],
                 'Sleep_Latency_Score': sleep.sleep_latency[date],
                 'Risk_Score': risk.risk[date],
                 'Risk_Appeal_Score': risk.risk_appeal[date],
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
            merged_df = merge_all_on_same_day(file_path, risk, sleep, mood)
            all_merged_dfs.append(merged_df)

        final_merged_df = pd.concat(all_merged_dfs, ignore_index=True)
        graph = Graph(final_merged_df, group)
        graph.mediation_analysis()


def main():
    get_all_correlations()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
