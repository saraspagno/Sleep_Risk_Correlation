import constants
from database import DataBase
from collections import defaultdict
import statistics


class Risk:
    """Risk parses the database to retrieve the risk scores and the accuracies scores.

    Attributes:
        db: a database instance.
        rows: the rows returned by the risk query.
        continuous_corr: map between date and a continuous risk score [0,100].
    """

    def __init__(self, self_db: DataBase):
        self.db = self_db
        self.rows = self.get_rows()
        self.image_outcomes = defaultdict(lambda: [])

        average, risk_appeal = self.parse()
        self.continuous_corr = average
        self.risk_appeal = risk_appeal

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
        all_risk_appeal = {}
        for r in self.rows:
            # taking all parameters needed from the row
            [r0, r1, choice, time, im0, im1] = [r[0], r[1], r[2], r[3], r[4], r[5]]
            [outcome, feedback, rank0, rank1] = [r[6], r[7], r[8], r[9]]
            if calculate_trial_scores(rank0, rank1, feedback, r0, r1):
                add_risk(r0, r1, choice, time, all_risks)
                self.add_risk_appeal(im0, im1, r0, r1, time, all_risk_appeal)
            if feedback:
                if choice == 0:
                    self.image_outcomes[im0].append(outcome)
                else:
                    self.image_outcomes[im1].append(outcome)

        day_to_average_risk_map = {}
        day_to_risk_appeal = {}

        # using each day to calculate the average (continuous) and majority (binary) score of the day
        for day, risks in all_risks.items():
            day_to_average_risk_map[day] = statistics.mean(risks)

        # calculating expected score of the day
        for day, expected in all_risk_appeal.items():
            day_to_risk_appeal[day] = statistics.mean(expected)

        return [day_to_average_risk_map, day_to_risk_appeal]

    def add_risk_appeal(self, im0: int, im1: int, r0: int, r1: int, time, all_expected: dict) -> None:
        average0, average1 = 0, 0
        if len(self.image_outcomes[im0]) != 0:
            average0 = statistics.mean(self.image_outcomes[im0])
        if len(self.image_outcomes[im1]) != 0:
            average1 = statistics.mean(self.image_outcomes[im1])

        # if the risky is image0
        if r0 > r1:
            # the appeal of picking the risky image is the difference in average return
            risk_appeal = average0 - average1
        else:
            risk_appeal = average1 - average0

        all_expected.setdefault(constants.to_unique_day(time), []).append(float(risk_appeal))


def calculate_trial_scores(rank0: int, rank1: int, feedback: int, r0: int, r1: int) -> bool:
    if rank0 != 1 or rank1 != 1 or feedback == 1 or r0 == r1:
        return False
    return True


def add_risk(r0: float, r1: float, choice: int, time, all_risks: dict):
    """Adds one couple of day:risk to the all risks map (in place, doesn't return anything).
    """
    if choice == 0:
        risk_taken = int(r0 > r1)
    else:
        risk_taken = int(r1 > r0)
    # appending key of day and value a list with risks taken between 0 and 100
    # each day has 70 trials scores
    all_risks.setdefault(constants.to_unique_day(time), []).append(float(risk_taken))
