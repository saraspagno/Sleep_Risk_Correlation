import constants
from database import DataBase
from collections import defaultdict
import statistics


class Risk:
    """Risk parses the database to retrieve the risk scores and the accuracies scores.
    """

    def __init__(self, self_db: DataBase):
        self.db = self_db
        self.rows = self.get_rows()
        # dictionary between image number and its scores
        self.image_outcomes = defaultdict(lambda: [])
        self.risk, self.risk_appeal = self.parse()

    def get_rows(self):
        """Reads the rows using the risk query.
        Returns: the rows returned by the query.
        """
        return self.db.execute(constants.RISK_QUERY)

    def parse(self) -> []:
        """Parses the rows into the map.
        """
        # map between a day and a risk or an accuracy
        all_risks = {}
        all_risk_appeal = {}
        for r in self.rows:
            # taking all parameters needed from the row
            [r0, r1, choice, time, im0, im1] = [r[0], r[1], r[2], r[3], r[4], r[5]]
            [outcome, feedback, rank0, rank1, trial] = [r[6], r[7], r[8], r[9], r[10]]
            if should_calculate_risk_score(rank0, rank1, feedback, r0, r1):
                self.add_risk(r0, r1, choice, time, trial, im0, im1, all_risks)
                self.add_risk_appeal(im0, im1, r0, r1, time, all_risk_appeal)
            if feedback:
                if choice == 0:
                    self.image_outcomes[im0].append(outcome)
                else:
                    self.image_outcomes[im1].append(outcome)

        day_to_average_risk_map = {}
        day_to_risk_appeal = {}

        # calculating the risk and risk appeal scores per day
        minimum_size_average = 5
        for day, risks in all_risks.items():
            if len(risks) >= minimum_size_average:
                day_to_average_risk_map[day] = statistics.mean(risks)
                day_to_risk_appeal[day] = statistics.mean(all_risk_appeal[day])

        # the return is a map day->risk, and a map day->risk_appeal
        return [day_to_average_risk_map, day_to_risk_appeal]

    def add_risk(self, r0: float, r1: float, choice: int, time, trial_n: int, im0: int, im1: int, all_risks: dict):
        """Adds one couple of day:risk to the all risks map (in place, doesn't return anything).
        """

        if choice == 0:
            # if r0 > r1 it means im0 is the risky one
            if r0 > r1:
                # therefore, risk is taken (score 1)
                risk_taken = 1 * len(self.image_outcomes[im0]) / 100
            else:
                # otherwise, im1 is the risky one, therefore risk is not taken (score -1)
                risk_taken = -1 * len(self.image_outcomes[im0]) / 100
        else:
            if r1 > r0:
                risk_taken = 1 * len(self.image_outcomes[im1]) / 100
            else:
                risk_taken = -1 * len(self.image_outcomes[im1]) / 100

        # appending key of day and value a list with risks taken between 0 and 100
        # each day has 70 trials scores
        all_risks.setdefault(constants.to_unique_day(time), []).append(float(risk_taken))

    def add_risk_appeal(self, im0: int, im1: int, r0: int, r1: int, time, all_expected: dict) -> None:
        average0, average1 = 0, 0
        if len(self.image_outcomes[im0]) != 0:
            # the average all outcomes given by image0 (in previous trials with feedback)
            average0 = statistics.mean(self.image_outcomes[im0])
        if len(self.image_outcomes[im1]) != 0:
            average1 = statistics.mean(self.image_outcomes[im1])

        # if the risky is image0
        if r0 > r1:
            # the appeal of picking the risky image (im0) is the difference in average return
            risk_appeal = average0 - average1
        else:
            risk_appeal = average1 - average0

        all_expected.setdefault(constants.to_unique_day(time), []).append(float(risk_appeal))


def should_calculate_risk_score(rank0: int, rank1: int, feedback: int, r0: int, r1: int) -> bool:
    # including this trial into risk_score and risk_appeal only if ranks are 1, no feedback, and different reward
    if rank0 != 1 or rank1 != 1 or feedback == 1 or r0 == r1:
        return False
    return True



