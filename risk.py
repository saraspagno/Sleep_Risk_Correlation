import constants
from scipy.stats import rv_discrete
from database import DataBase
from collections import defaultdict


class Risk:
    """Risk parses the database to retrieve the risk scores.

    Attributes:
        db: a database instance.
        rows: the rows returned by the risk query.
        continuous_corr: map between date and a continuous risk score [0,100].
        binary_corr: map between date and a binary risk score 0 or 1.
    """

    def __init__(self, self_db: DataBase):
        self.db = self_db
        self.rows = self.get_rows()
        if constants.USE_PERCEIVED_REWARD:
            # format is [reward prob, neutral prob, punishment prob]
            self.perceived_risk = defaultdict(lambda: [1, 1, 1])
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
        Returns: map between date and risk scores.
        """
        result = {}
        for r in self.rows:
            [r0, r1, choice, time, im0, im1, outcome] = [r[0], r[1], r[2], r[3], r[4], r[5], r[6]]
            # calculating the risk taken using helper function
            if not constants.USE_PERCEIVED_REWARD:
                risk_taken = get_objective_risk_taken(r0, r1, choice)
            else:
                risk_taken = self.get_perceived_risk_taken(im0, im1, choice, outcome)
            # skipping case in which no choice was made
            if risk_taken != -1:
                # appending key of day and value a list with risks taken between 0 and 100
                # each day has 70 trials scores
                result.setdefault(constants.to_unique_day(time), []).append(float(risk_taken * 100))

        average = {}
        majority = {}

        # using each day to calculate the average (continuous) and majority (binary) score of the day
        for day, values in result.items():
            av = sum(values) / len(values)
            average[day] = av
            majority_above_half = sum(1 for v in values if v > 50) >= len(values) / 2
            majority[day] = 1 if majority_above_half else 0
        return [average, majority]

    def get_perceived_risk_taken(self, im0: int, im1: int, choice: int, outcome: int) -> float:
        # if image not in dictionary, assuming same probability (1/3) for each outcome of win, neutral, loose.
        probs0 = get_normalized_probs(self.perceived_risk[im0])
        probs1 = get_normalized_probs(self.perceived_risk[im1])
        outcomes = [1, 0, -1]
        # creating two random variables with probabilities for image0 and image1
        var0 = rv_discrete(values=(outcomes, probs0)).var()
        var1 = rv_discrete(values=(outcomes, probs1)).var()
        total_variance = var0 + var1
        perceived_risk_0 = var0 / total_variance
        perceived_risk_1 = var1 / total_variance

        if abs(perceived_risk_0 - perceived_risk_1) < 0.1:
            risk_taken = -1
        elif choice == 0:
            risk_taken = int(perceived_risk_0 > perceived_risk_1)
        else:
            risk_taken = int(perceived_risk_1 > perceived_risk_0)

        if choice == 0:
            self.update_probabilities(im0, outcome)
        else:
            self.update_probabilities(im1, outcome)

        return risk_taken

    def update_probabilities(self, im: int, outcome: int):
        probs = self.perceived_risk[im]
        if outcome == 1:
            # if the outcome was reward, we increase the probability of reward
            probs[0] += 1
        elif outcome == 0:
            probs[1] += 1
        else:
            probs[2] += 1


def get_normalized_probs(probs: list):
    total = sum(probs)
    actual_probs = [p / total for p in probs]
    return actual_probs


def get_objective_risk_taken(prob0: int, prob1: int, choice: int):
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
