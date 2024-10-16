import constants
from scipy.stats import rv_discrete

from accuracy import Accuracy
from database import DataBase
from collections import defaultdict
import statistics


class Risk:
    """Risk parses the database to retrieve the risk scores and the accuracies scores.

    Attributes:
        db: a database instance.
        rows: the rows returned by the risk query.
        continuous_corr: map between date and a continuous risk score [0,100].
        binary_corr: map between date and a binary risk score 0 or 1.
    """

    def __init__(self, self_db: DataBase):
        self.db = self_db
        self.rows = self.get_rows()
        self.image_outcomes = defaultdict(lambda: [])

        average, majority, expected = self.parse()
        self.continuous_corr = average
        self.binary_corr = majority
        self.expected_corr = expected

    def get_rows(self):
        """Reads the rows using the risk query.
        Returns: the rows returned by the query.
        """
        return self.db.execute(constants.RISK_QUERY)

    def parse(self) -> []:
        """Parses the rows into the map.
        :returns: 3 maps, continuous_corr, binary_corr, and expected_corr described above.
        """
        # map between a day and a risk or an accuracy
        all_risks = {}
        all_expected = {}
        for r in self.rows:
            # taking all parameters needed from the row
            [r0, r1, choice, time, im0, im1] = [r[0], r[1], r[2], r[3], r[4], r[5]]
            [outcome, feedback, rank0, rank1] = [r[6], r[7], r[8], r[9]]
            if calculate_trial_scores(rank0, rank1, feedback, r0, r1):
                add_risk(r0, r1, choice, time, all_risks)
                self.add_expected(im0, im1, r0, r1, time, all_expected)
            if feedback:
                if choice == 0:
                    self.image_outcomes[im0].append(outcome)
                else:
                    self.image_outcomes[im1].append(outcome)

        day_to_average_risk_map = {}
        day_to_binary_risk_map = {}
        day_to_expected_map = {}

        # using each day to calculate the average (continuous) and majority (binary) score of the day
        for day, risks in all_risks.items():
            day_to_average_risk_map[day] = statistics.mean(risks)
            majority_above_half = sum(1 for v in risks if v > 50) >= len(risks) / 2
            day_to_binary_risk_map[day] = 1 if majority_above_half else 0

        # calculating expected score of the day
        for day, expected in all_expected.items():
            day_to_expected_map[day] = statistics.mean(expected)

        return [day_to_average_risk_map, day_to_binary_risk_map, day_to_expected_map]

    def add_expected(self, im0: int, im1: int, r0: int, r1: int, time, all_expected: dict) -> None:
        if len(self.image_outcomes[im0]) == 0:
            average0 = 0
        else:
            average0 = statistics.mean(self.image_outcomes[im0])

        if len(self.image_outcomes[im1]) == 0:
            average1 = 0
        else:
            average1 = statistics.mean(self.image_outcomes[im1])

        if r0 > r1:
            expected = average1 - average0
        else:
            expected = average0 - average1

        all_expected.setdefault(constants.to_unique_day(time), []).append(float(expected))


def calculate_trial_scores(rank0: int, rank1: int, feedback: int, r0: int, r1: int) -> bool:
    if rank0 != 1 or rank1 != 1 or feedback == 1 or r0 == r1:
        return False
    return True


def  add_risk(r0: float, r1: float, choice: int, time, all_risks: dict):
    """Adds one couple of day:risk to the all risks map (in place, doesn't return anything).
    """
    # calculating the risk taken using helper function
    if choice == 0:
        risk_taken = int(r0 > r1)
    else:
        risk_taken = int(r0 > r1)
    # appending key of day and value a list with risks taken between 0 and 100
    # each day has 70 trials scores
    all_risks.setdefault(constants.to_unique_day(time), []).append(float(risk_taken))
