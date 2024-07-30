import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr


class Graph:
    def __init__(self, binary: dict, continuous: dict, group: str):
        self.binary = binary
        self.continuous = continuous
        self.group = group

    def show_regression(self):
        data = {'sleep': list(self.continuous.keys()),
                'risk': list(self.continuous.values())}
        df = pd.DataFrame(data)
        r_value, p_value = pearsonr(df['sleep'], df['risk'])
        plt.figure(figsize=(10, 6))
        sns.regplot(x='sleep', y='risk', data=df)
        plt.text(0.1, 0.9, f'r = {r_value:.2f} \nP-value: {p_value:.4f}', transform=plt.gca().transAxes)
        plt.title(f'Regression between Sleep and Risk, Group: {self.group}')
        plt.xlabel('Sleep (quality)')
        plt.ylabel('Risk (value)')
        plt.show()

    def show_box_plot(self):
        data = {'sleep': list(self.binary.keys()),
                'risk': list(self.binary.values())}
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='risk', y='sleep', data=df)
        plt.title(f'Box Plot of Sleep vs. Risk (Binary), Group: {self.group}')
        plt.xlabel('Risk (Binary)')
        plt.ylabel('Sleep (Quality)')
        plt.show()
