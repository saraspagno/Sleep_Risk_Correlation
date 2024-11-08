import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
import pingouin as pg
import statsmodels.api as sm


class Graph:
    def __init__(self, merged_df, group: str):
        self.group = group
        self.merged_df = merged_df

    def show_risk_risk_appeal_regression(self):
        if len(self.merged_df) < 3:
            return
        r_value, p_value = pearsonr(self.merged_df['Risk Appeal Score'], self.merged_df['Risk Score'])
        plt.figure(figsize=(10, 6))
        sns.regplot(x='Risk Appeal Score', y='Risk Score', data=self.merged_df)
        plt.text(0.1, 0.9, f'R-value = {r_value:.2f} \nP-value: {p_value:.4f}', transform=plt.gca().transAxes)
        plt.title(f'Regression of Risk vs. Risk Appeal Score, Group: {self.group}')
        plt.xlabel('Risk Appeal Score')
        plt.ylabel('Daily Risk Taken')
        plt.show()

    def tests(self, sleep_score: str):
        # Pearson correlation sleep / risk
        simple_correlation = self.merged_df[[sleep_score, 'Risk Score']].corr(method='pearson')
        print(f"Group: {self.group}, Pearson Correlation\n{simple_correlation}\n")

        # Spearman correlation sleep / risk
        simple_correlation = self.merged_df[[sleep_score, 'Risk Score']].corr(method='spearman')
        print(f"Group: {self.group}, Spearman Correlation\n{simple_correlation}\n\n")

        # Multi-variate Regression model
        X = self.merged_df[[sleep_score, 'Risk Appeal Score']]
        X = sm.add_constant(X)
        y = self.merged_df['Risk Score']
        model = sm.OLS(y, X).fit()
        print(f"Group: {self.group}, Multi-variate Regression Model\n{model.summary()}\n\n\n")

        #  Interaction Analysis
        self.merged_df['Interaction'] = self.merged_df[sleep_score] * self.merged_df['Risk Appeal Score']
        X_interaction = self.merged_df[[sleep_score, 'Risk Appeal Score', 'Interaction']]
        X_interaction = sm.add_constant(X_interaction)
        model_interaction = sm.OLS(y, X_interaction).fit()
        print(f"Group: {self.group}, Interaction Analysis\n{model_interaction.summary()}\n\n\n")

    def show_regression(self, sleep_score: str):
        if len(self.merged_df) < 3:
            return
        r_value, p_value = pearsonr(self.merged_df[sleep_score], self.merged_df['Risk Score'])
        plt.figure(figsize=(10, 6))
        sns.regplot(x=sleep_score, y='Risk Score', data=self.merged_df)
        plt.text(0.1, 0.9, f'R-value = {r_value:.2f} \nP-value: {p_value:.4f}', transform=plt.gca().transAxes)
        plt.title(f'Regression of Risk vs. {sleep_score}, Group: {self.group}')
        plt.xlabel('Daily Sleep Quality')
        plt.ylabel('Daily Risk Taken')
        plt.show()
