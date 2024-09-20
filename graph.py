import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr


class Graph:
    def __init__(self, risk_binary: dict, sleep_risk_continuous: dict, day_accuracy: dict, group: str):
        self.sleep_risk_binary = risk_binary
        self.sleep_risk_continuous = sleep_risk_continuous
        self.day_accuracy = day_accuracy
        self.group = group

    def show_sleep_risk_regression(self):
        data = {'sleep': list(self.sleep_risk_continuous.keys()),
                'risk': list(self.sleep_risk_continuous.values())}
        df = pd.DataFrame(data)
        r_value, p_value = pearsonr(df['sleep'], df['risk'])
        plt.figure(figsize=(10, 6))
        sns.regplot(x='sleep', y='risk', data=df)
        plt.text(0.1, 0.9, f'R-value = {r_value:.2f} \nP-value: {p_value:.4f}', transform=plt.gca().transAxes)
        plt.title(f'Regression Sleep vs. Risk, Group: {self.group}')
        plt.xlabel('Daily Sleep Quality')
        plt.ylabel('Daily Risk taken')
        plt.show()

    def show_risk_sleep_box_plot(self):
        data = {'sleep': list(self.sleep_risk_binary.keys()),
                'risk': list(self.sleep_risk_binary.values())}
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='risk', y='sleep', data=df)
        plt.title(f'Box Plot Sleep vs. Risk, Group: {self.group}')
        plt.xlabel('Daily Risk Taken')
        plt.ylabel('Daily Sleep Quality')
        plt.show()

    def show_accuracy_day_regression(self):
        days = list(self.day_accuracy.keys())
        accuracies = list(self.day_accuracy.values())
        plt.figure(figsize=(8, 6))
        plt.plot(days, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
        plt.xlabel('Day')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy over time, Group: {self.group}')
        plt.xticks(ticks=range(len(days)), labels=['' for _ in days])
        plt.xticks(days)
        plt.legend()
        plt.show()
