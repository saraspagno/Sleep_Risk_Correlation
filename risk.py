import constants
from scipy.stats import rv_discrete

from accuracy import Accuracy
from database import DataBase
from collections import defaultdict


class Risk:
    """Risk parses the database to retrieve the risk scores and the accuracies scores.

    Attributes:
        db: a database instance.
        rows: the rows returned by the risk query.
        continuous_corr: map between date and a continuous risk score [0,100].
        binary_corr: map between date and a binary risk score 0 or 1.
        accuracy_corr: map between date and accuracy score [0,100].
    """

    def __init__(self, self_db: DataBase, accuracy: Accuracy):
        self.db = self_db
        self.rows = self.get_rows()
        self.perceived_risk = defaultdict(lambda: [1.0, 1.0, 1.0])
        self.accuracy = accuracy
        average, majority = self.parse()
        self.continuous_corr = average
        self.binary_corr = majority

    def get_rows(self):
        """Reads the rows using the risk query.
        Returns: the rows returned by the query.
        """
        return self.db.execute(constants.RISK_QUERY)

    def parse(self) -> []:
        """Parses the rows into the map.
        :returns: 3 maps, continuous_corr, binary_corr, and accuracy_corr as described above.
        """
        # map between a day and a risk or an accuracy
        all_risks = {}
        for r in self.rows:
            # taking all parameters needed from the row
            [r0, r1, p0, p1, choice, time, im0, im1] = [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]]
            [outcome, feedback, rank0, rank1, block] = [r[8], r[9], r[10], r[11], r[12]]
            self.accuracy.add_accuracy(r0, r1, p0, p1, self.perceived_risk[im0], self.perceived_risk[im1], choice,
                                       block, feedback)
            self.add_risk(r0, r1, choice, time, self.perceived_risk[im0], self.perceived_risk[im1], outcome, feedback, rank0, rank1, all_risks)

        day_to_average_risk_map = {}
        day_to_binary_risk_map = {}

        # TODO something with accuracies?

        # using each day to calculate the average (continuous) and majority (binary) score of the day
        for day, risks in all_risks.items():
            av = sum(risks) / len(risks)
            day_to_average_risk_map[day] = av
            majority_above_half = sum(1 for v in risks if v > 50) >= len(risks) / 2
            day_to_binary_risk_map[day] = 1 if majority_above_half else 0

        return [day_to_average_risk_map, day_to_binary_risk_map]

    def add_risk(self, r0: float, r1: float, choice: int, time, perceived0: [], perceived1: [], outcome: int, feedback: int,
                 rank0: int, rank1: int,
                 all_risks: dict):
        """Adds one couple of day:risk to the all risks map (in place, doesn't return anything).
        """
        if constants.SKIP_IF_RANK_NOT_1 and (rank0 != 1 or rank1 != 1):
            return
        probs0, probs1 = constants.get_normalized_probs(perceived0), constants.get_normalized_probs(perceived1)
        # calculating the risk taken using helper function
        if not constants.USE_PERCEIVED_REWARD:
            risk_taken = get_objective_risk_taken(r0, r1, choice, rank0, rank1, probs0, probs1)
        else:
            risk_taken = self.get_perceived_risk_taken(probs0, probs1, choice)
            # updating the perceived probabilities if feedback was provided
        if feedback:
            self.update_probabilities(perceived0, perceived1, outcome, choice)

        # skipping case in which no significant change between choices
        if risk_taken != -1:
            # appending key of day and value a list with risks taken between 0 and 100
            # each day has 70 trials scores
            all_risks.setdefault(constants.to_unique_day(time), []).append(float(risk_taken * 100))

    def get_perceived_risk_taken(self, probs0: [], probs1: [], choice: int) -> int:
        """ calculates the risk using the "perceived" probabilities.
        :returns: the risk taken
         """
        outcomes = [1, 0, -1]
        # creating two random variables with probabilities for image0 and image1
        var0 = rv_discrete(values=(outcomes, probs0)).var()
        var1 = rv_discrete(values=(outcomes, probs1)).var()
        total_variance = var0 + var1
        perceived_risk_0 = var0 / total_variance
        perceived_risk_1 = var1 / total_variance

        perceived_risk_0 *= probs0[2]
        perceived_risk_1 *= probs1[2]

        # if the difference between the risk in the two images is too small, we do not include it
        if abs(perceived_risk_0 - perceived_risk_1) <= 0.1:
            risk_taken = -1
        elif choice == 0:
            # returning 1 (risk taken) if the risk of choice0 was higher than the risk of choice1
            risk_taken = int(perceived_risk_0 > perceived_risk_1)
        else:
            risk_taken = int(perceived_risk_1 > perceived_risk_0)

        return risk_taken

    def update_probabilities(self, perceived0: [], perceived1: [], outcome: int, choice: int):
        """Adds one couple of day:risk to the all risks map (in place, doesn't return anything) using bayes.
        """

        if choice == 0:
            probs = perceived0
        else:
            probs = perceived1

        update_increase = 1
        if outcome == 1:
            # if the outcome was reward, we increase the probability of reward
            probs[0] += update_increase
        elif outcome == 0:
            probs[1] += update_increase
        else:
            probs[2] += update_increase


def get_objective_risk_taken(r0: float, r1: float, choice: int, rank0: int, rank1: int, probs0: [], probs1: []) -> int:
    """Calculates the risk taken given the probabilities and the choice.
    Args:
      r0: percentages of reward for image1, 0.15 is safe and 0.5 is risky.
      r1: percentages of reward for image2.
      choice: the choice taken by the volunteer.
      rank0: if 1, then the images are safe / risky, we exclude images which of different types.
      rank1: if 1, then the images are safe / risky, we exclude images which of different types.
    Returns: the risk taken.
    """

    if rank0 != 1 or rank1 != 1:
        return -1
    elif r1 == r0:
        return -1
    # elif abs(probs0[0] - r0) > 0.15 or abs(probs1[0] - r1) > 0.15:
    #     print("Too much difference")
        return -1
    elif choice == 0:
        return int(r0 > r1)
    else:
        return int(r0 > r1)
