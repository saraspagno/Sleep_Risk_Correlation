import constants
import re
from database import DataBase


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
        self.overall, self.woke_early, self.woke_many_time, self.sleep_latency = self.parse()

    def get_rows(self):
        """Reads the rows using the sleep query.
        Returns: the rows returned by the query.
        """
        return self.db.execute(constants.SLEEP_QUERY)

    def parse(self):
        """Parses the rows into the map.
        Returns: map between date and sleep score.
        """
        overall = {}
        woke_early = {}
        woke_many_time = {}
        sleep_latency = {}
        for r in self.rows:
            # regex for retrieve the exact sleep score
            overall_score = re.search(r'overall=(\d+)', r[0]).group(1)
            woke_early_score = re.search(r'woke early=(\d+)', r[0]).group(1)
            woke_many_time_score = re.search(r'woke many times=(\d+)', r[0]).group(1)
            sleep_latency_score = re.search(r'sleep latency=(\d+)', r[0]).group(1)
            # setting key of date and value of sleep score
            overall[constants.to_unique_day(r[1])] = float(overall_score)
            woke_early[constants.to_unique_day(r[1])] = float(woke_early_score)
            woke_many_time[constants.to_unique_day(r[1])] = float(woke_many_time_score)
            sleep_latency[constants.to_unique_day(r[1])] = float(sleep_latency_score)
        return [overall, woke_early, woke_many_time, sleep_latency]
