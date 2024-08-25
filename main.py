import os
import sqlite3
import re
from datetime import datetime
from scipy.stats import rv_discrete

import constants
from graph import Graph


class DataBase:
    """DataBase class holds a database instance.

    Attributes:
        cursor: used for executing queries.
    """

    def __init__(self, db_file_name: str):
        """Initializes the database instance.
        Args:
          db_file_name: the name of the database file.
        """
        conn = sqlite3.connect(db_file_name)
        self.cursor = conn.cursor()

    def execute(self, query: str):
        """Executes one query on the database.
        Args:
          query: tstring query to execute.
        Returns: the rows returned by the query.
        """
        return self.cursor.execute(query).fetchall()


class Sleep:
    """Sleep parses the database to retrieve the sleep scores.

    Attributes:
        db: a database instance.
        rows: the rows returned by the sleep query.
        correlations: map between date and sleep score.
    """

    def __init__(self, self_db: DataBase):
        """Initializes the sleep instance.
        Args:
          self_db: the database instance.
        """
        self.db = self_db
        self.rows = self.get_rows()
        self.correlations = self.parse()

    def get_rows(self):
        """Reads the rows using the sleep query.
        Returns: the rows returned by the query.
        """
        return self.db.execute(constants.SLEEP_QUERY)

    def parse(self):
        """Parses the rows into the map.
        Returns: map between date and sleep score.
        """
        result = {}
        for r in self.rows:
            # regex for retrieve the exact sleep score
            score = re.search(r'overall=(\d+)', r[0]).group(1)
            # setting key of date and value of sleep score
            result[to_unique_day(r[1])] = float(score)
        return result


class Risk:
    """Risk parses the database to retrieve the risk scores.

    Attributes:
        db: a database instance.
        rows: the rows returned by the risk query.
        continuous_corr: map between date and a continuous risk score [0,100].
        binary_corr: map between date and a binary risk score 0 or 1.
    """

    def __init__(self, self_db):
        self.db = self_db
        self.rows = self.get_rows()
        average, majority = self.parse()
        self.continuous_corr = average
        self.binary_corr = majority

    def get_rows(self):
        """Reads the rows using the risk query.
        Returns: the rows returned by the query.
        """
        return self.db.execute(constants.RISK_QUERY)

    def parse(self):
        """Parses the rows into the map.
        Returns: map between date and sleep score.
        """
        result = {}
        for r in self.rows:
            # calculating the risk taken using helper function
            risk_taken = get_risk_taken(r[0], r[1], r[2])
            # skipping case in which no choice was made
            if risk_taken != -1:
                # appending key of day and value a list with risks taken between 0 and 100
                # each day has 70 trials scores
                result.setdefault(to_unique_day(r[3]), []).append(float(risk_taken * 100))

        average = {}
        majority = {}

        # using each day to calculate the average (continuous) and majority (binary) score of the day
        for day, values in result.items():
            av = sum(values) / len(values)
            average[day] = av
            majority_above_half = sum(1 for v in values if v > 50) >= len(values) / 2
            majority[day] = 1 if majority_above_half else 0
        return [average, majority]


def get_risk_taken(prob0: int, prob1: int, choice: int):
    """Calculates the risk taken given the probabilities and the choice.
    Args:
      prob0: percentages of reward for image1, 0.15 is safe and 0.5 is risky.
      prob1: percentages of reward for image2.
      choice: the choice taken by the volunteer.
    Returns: the risk taken.
    """
    if prob0 == prob1:
        return -1
    if choice == 0:
        return int(prob0 > prob1)
    else:
        return int(prob1 > prob0)


def to_unique_day(timestamp):
    """Turns a timestamp into a unique day without seconds.
    Args:
      timestamp: the timestamp which might include seconds.
    Returns: a unique string representing one day, excluding time and seconds.
    """
    original_timestamp_seconds = timestamp / 1000
    datetime_obj = datetime.fromtimestamp(original_timestamp_seconds)
    return datetime_obj.strftime("%d %B %Y")


def merge_sleep_risk_on_date(sleep_values: dict, risk_values: dict):
    """Turns a timestamp into a unique day without seconds.
    Args:
        sleep_values: the map of date and sleep score.
        risk_values: the map of date and risk score.
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
        print(f"processing file: {filename}")
        file_path = os.path.join(group, filename)
        db = DataBase(file_path)
        risk = Risk(db)
        sleep = Sleep(db)
        sleep_w_risk_binary_all.update(merge_sleep_risk_on_date(sleep.correlations, risk.binary_corr))
        sleep_w_risk_continuous_all.update(merge_sleep_risk_on_date(sleep.correlations, risk.continuous_corr))
    return sleep_w_risk_binary_all, sleep_w_risk_continuous_all


if __name__ == '__main__':
    binary, continuous = get_all_correlations(constants.GROUP)
    graph = Graph(binary, continuous, constants.GROUP)
    graph.show_regression()
    graph.show_box_plot()
