import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr


class Graph:
    def __init__(self, risk_binary: dict, sleep_risk_continuous: dict, time_accuracy: dict, group: str):
        self.sleep_risk_binary = risk_binary
        self.sleep_risk_continuous = sleep_risk_continuous
        self.time_accuracy = time_accuracy
        self.group = group

    def show_sleep_risk_regression(self):
        if len(self.sleep_risk_continuous) < 3:
            return
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

    def show_accuracy_time_regression(self):
        time = list(self.time_accuracy.keys())
        accuracies = list(self.time_accuracy.values())
        plt.figure(figsize=(8, 6))
        plt.plot(time, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy over time, Group: {self.group}')
        plt.legend()
        plt.show()
