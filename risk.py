import constants
from scipy.stats import rv_discrete
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

    def __init__(self, self_db: DataBase):
        self.db = self_db
        self.rows = self.get_rows()
        if constants.USE_PERCEIVED_REWARD:
            # format is [reward prob, neutral prob, punishment prob]
            self.perceived_risk = defaultdict(lambda: [1, 1, 1])
        average, majority, accuracies = self.parse()
        self.continuous_corr = average
        self.binary_corr = majority
        self.accuracy_corr = accuracies

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
        all_accuracies = {}
        for r in self.rows:
            # taking all parameters needed from the row
            [r0, r1, p0, p1, choice, time, im0, im1] = [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]]
            [outcome, feedback, rank0, rank1] = [r[8], r[9], r[10], r[11]]
            self.add_risk(r0, r1, choice, time, im0, im1, outcome, feedback, rank0, rank1, all_risks)
            self.add_accuracy(r0, r1, p0, p1, im0, im1, choice, time, all_accuracies)

        day_to_average_risk_map = {}
        day_to_binary_risk_map = {}
        day_to_accuracy_map = {}

        # using each day to calculate the average accuracy of the day
        for day, accuracies in all_accuracies.items():
            av = sum(accuracies) / len(accuracies)
            day_to_accuracy_map[day] = av

        # using each day to calculate the average (continuous) and majority (binary) score of the day
        for day, risks in all_risks.items():
            # if the flag to account for accuracy is on, AND:
            # either the day is not in the accuracy map (meaning we have not enough data for it)
            # or either the accuracy for the day didn't reach the threshold
            # then - we don't skip the risk calculation, meaning we don't count this day.
            if constants.ACCOUNT_ACCURACY and ((day not in day_to_accuracy_map) or day_to_accuracy_map[day] < 50):
                continue
            av = sum(risks) / len(risks)
            day_to_average_risk_map[day] = av
            majority_above_half = sum(1 for v in risks if v > 50) >= len(risks) / 2
            day_to_binary_risk_map[day] = 1 if majority_above_half else 0

        return [day_to_average_risk_map, day_to_binary_risk_map, day_to_accuracy_map]

    def add_risk(self, r0: float, r1: float, choice: int, time, im0: int, im1: int, outcome: int, feedback: int,
                 rank0: int, rank1: int,
                 all_risks: dict):
        """Adds one couple of day:risk to the all risks map (in place, doesn't return anything).
        """

        # calculating the risk taken using helper function
        if not constants.USE_PERCEIVED_REWARD:
            risk_taken = get_objective_risk_taken(r0, r1, choice, rank0, rank1)
        else:
            risk_taken = self.get_perceived_risk_taken(im0, im1, choice)
            # updating the perceived probabilities if feedback was provided
            if feedback:
                self.update_probabilities(im0, im1, outcome, choice)

        # skipping case in which no significant change between choices
        if risk_taken != -1:
            # appending key of day and value a list with risks taken between 0 and 100
            # each day has 70 trials scores
            all_risks.setdefault(constants.to_unique_day(time), []).append(float(risk_taken * 100))

    def get_perceived_risk_taken(self, im0: int, im1: int, choice: int) -> int:
        """ calculates the risk using the "perceived" probabilities.
        :returns: the risk taken
         """

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

        # if the difference between the risk in the too images is too small, we do not include it
        if abs(perceived_risk_0 - perceived_risk_1) < 0.1:
            risk_taken = -1
        elif choice == 0:
            # returning 1 (risk taken) if the risk of choice0 was higher than the risk of choice1
            risk_taken = int(perceived_risk_0 > perceived_risk_1)
        else:
            risk_taken = int(perceived_risk_1 > perceived_risk_0)

        return risk_taken

    def update_probabilities(self, im0: int, im1: int, outcome: int, choice: int):
        """Adds one couple of day:risk to the all risks map (in place, doesn't return anything) using bayes.
        """

        if choice == 0:
            probs = self.perceived_risk[im0]
        else:
            probs = self.perceived_risk[im1]

        if outcome == 1:
            # if the outcome was reward, we increase the probability of reward
            probs[0] += 1
        elif outcome == 0:
            probs[1] += 1
        else:
            probs[2] += 1

    def add_accuracy(self, r0: float, r1: float, p0: float, p1: float, im0: int, im1: int, choice: int, time,
                     all_accuracies: dict):
        """Adds one couple of day:accuracy to the all the accuracies map (in place, doesn't return anything).
        """

        if constants.USE_PERCEIVED_REWARD:
            probs0 = get_normalized_probs(self.perceived_risk[im0])
            probs1 = get_normalized_probs(self.perceived_risk[im1])
        else:
            probs0 = [r0, 1 - r0 - p0, p0]
            probs1 = [r1, 1 - r1 - p1, p1]

        outcomes = [1, 0, -1]
        exp0 = rv_discrete(values=(outcomes, probs0)).expect()
        exp1 = rv_discrete(values=(outcomes, probs1)).expect()
        if abs(exp0 - exp1) < 0.1:
            accuracy = -1
        elif choice == 0:
            accuracy = int(exp0 > exp1)
        else:
            accuracy = int(exp1 > exp0)

        if accuracy != -1:
            all_accuracies.setdefault(constants.to_unique_day(time), []).append(float(accuracy * 100))


def get_normalized_probs(probs: list) -> []:
    """Takes the perceived probabilities values and normalizes them with values between 0 and 1.
    """
    total = sum(probs)
    actual_probs = [p / total for p in probs]
    return actual_probs


def get_objective_risk_taken(prob0: float, prob1: float, choice: int, rank0: int, rank1: int) -> int:
    """Calculates the risk taken given the probabilities and the choice.
    Args:
      prob0: percentages of reward for image1, 0.15 is safe and 0.5 is risky.
      prob1: percentages of reward for image2.
      choice: the choice taken by the volunteer.
      rank0: if 1, then the images are safe / risky, we exclude images which of different types.
      rank1: if 1, then the images are safe / risky, we exclude images which of different types.
    Returns: the risk taken.
    """
    if rank0 != 1 or rank1 != 1 or prob0 == prob1:
        return -1
    if choice == 0:
        return int(prob0 > prob1)
    else:
        return int(prob1 > prob0)
