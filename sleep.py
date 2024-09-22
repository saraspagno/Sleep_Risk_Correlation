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
            result[constants.to_unique_day(r[1])] = float(score)
        return result
