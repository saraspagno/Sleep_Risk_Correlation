import statistics

from sklearn.preprocessing import MinMaxScaler
import constants
from database import DataBase
import re
from collections import defaultdict


class Mood:
    def __init__(self, self_db: DataBase):
        self.db = self_db
        self.rows = self.get_rows()
        self.valence, self.arousal, self.anxious, self.elated, self.sad, self.irritable, self.energetic = self.parse()

    def get_rows(self):
        return self.db.execute(constants.MOOD_QUERY)

    def parse(self) -> []:
        valence = defaultdict(list)
        arousal = defaultdict(list)
        anxious = defaultdict(list)
        elated = defaultdict(list)
        sad = defaultdict(list)
        irritable = defaultdict(list)
        energetic = defaultdict(list)
        scaler = MinMaxScaler(feature_range=(0, 100))

        for r in self.rows:
            valence_score = re.search(r'Valence=(\d+)', r[0])
            if valence_score:
                normalized = scaler.fit_transform([[float(valence_score.group(1))], [-300], [300]])[0][0]
                valence[constants.to_unique_day(r[1])].append(normalized)

            arousal_score = re.search(r'Arousal=(\d+)', r[0])
            if arousal_score:
                normalized = scaler.fit_transform([[float(arousal_score.group(1))], [-300], [300]])[0][0]
                arousal[constants.to_unique_day(r[1])].append(normalized)

            anxious_score = re.search(r'Anxious=(\d+)', r[0])
            if anxious_score:
                anxious[constants.to_unique_day(r[1])].append(float(anxious_score.group(1)))

            elated_score = re.search(r'Elated=(\d+)', r[0])
            if elated_score:
                elated[constants.to_unique_day(r[1])].append(float(elated_score.group(1)))

            sad_score = re.search(r'Sad=(\d+)', r[0])
            if sad_score:
                sad[constants.to_unique_day(r[1])].append(float(sad_score.group(1)))

            irritable_score = re.search(r'Irritable=(\d+)', r[0])
            if irritable_score:
                irritable[constants.to_unique_day(r[1])].append(float(irritable_score.group(1)))

            energetic_score = re.search(r'Energetic=(\d+)', r[0])
            if energetic_score:
                energetic[constants.to_unique_day(r[1])].append(float(energetic_score.group(1)))

        return [average(valence), average(arousal), average(anxious), average(elated), average(sad), average(irritable),
                average(energetic)]


def average(score_dict):
    result = {}
    for day, scores in score_dict.items():
        result[day] = statistics.mean(scores)
    return result
