from matplotlib import pyplot as plt
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
        if not feedback:
            return 0

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
            self.all_accuracies.setdefault(block, []).append(float(accuracy * 100))
        return accuracy

    def get_block_sliding_windows_accuracies(self):
        window_size = 10
        block_percentages = {}  # Store percentages for each block
        num_blocks = len(self.all_accuracies)

        # Step 1: Calculate the sliding window percentage of correct responses for each block
        for block, accuracies in self.all_accuracies.items():
            block_percentages[block] = []
            for start in range(len(accuracies) - window_size + 1):
                window = accuracies[start:start + window_size]  # Get the window of size 10
                correct_count = window.count(100)  # Count how many correct (100s) in this window
                percentage_correct = (correct_count / window_size) * 100  # Calculate percentage
                block_percentages[block].append(percentage_correct)

        # Step 2: Calculate the average correct percentage for each sliding window position across all blocks
        window_count = len(next(iter(block_percentages.values())))  # Number of windows (same for each block)
        window_averages = {}  # Store the average percentage for each sliding window start position

        for start in range(window_count):
            window_sum = 0
            for block in block_percentages:
                if start < len(block_percentages[block]):
                    window_sum += block_percentages[block][start]  # Sum the percentages at the current window
            window_averages[start + 1] = window_sum / num_blocks  # Calculate the average percentage for this window

        # Step 3: Plot the results (sliding window number vs. average correct percentage)
        plt.figure(figsize=(8, 5))

        # X-axis: Sliding window start position (1, 2, ..., based on window_count)
        x_values = list(window_averages.keys())

        # Y-axis: Average correct percentage
        y_values = list(window_averages.values())

        plt.plot(x_values, y_values, marker='o', linestyle='-', label='Average Percentage Across Blocks')

        # Set plot titles and labels
        plt.title('Average Sliding Window Correct Percentage Across All Blocks')
        plt.xlabel('Sliding Window Start Position')
        plt.ylabel('Average Correct Percentage (%)')

        # Optionally add a grid
        plt.grid(True)

        # Show the plot
        plt.show()

    def average_accuracies(self):
        time_to_accuracy_map = {}
        for time, accuracies in self.all_accuracies.items():
            av = sum(accuracies) / len(accuracies)
            time_to_accuracy_map[time] = av
        self.accuracy_corr = time_to_accuracy_map
