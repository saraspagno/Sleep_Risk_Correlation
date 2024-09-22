from scipy.stats import rv_discrete

import constants


class Accuracy:
    def __init__(self):
        self.all_accuracies = {}
        self.accuracy_corr = {}
        self.counter = 0

    def add_accuracy(self, r0: float, r1: float, p0: float, p1: float, perceived0: [], perceived1: [],
                     choice: int, block: int,
                     feedback: int) -> int:
        """Adds one couple of day:accuracy to the all the accuracies map (in place, doesn't return anything).
        """
        probs0, probs1 = constants.get_normalized_probs(perceived0), constants.get_normalized_probs(perceived1)

        if not constants.USE_PERCEIVED_REWARD:
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
            slide = 5
            for i in range(max(slide, self.counter - 100 + slide), self.counter + slide):
                self.all_accuracies.setdefault(i, []).append(float(accuracy * 100))
            self.counter += 1

        return accuracy




    def average_accuracies(self):
        time_to_accuracy_map = {}
        for time, accuracies in self.all_accuracies.items():
            av = sum(accuracies) / len(accuracies)
            time_to_accuracy_map[time] = av
        self.accuracy_corr = time_to_accuracy_map
