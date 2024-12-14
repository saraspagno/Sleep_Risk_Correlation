import math
from statistics import mean

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import stats
import pingouin as pg
import statsmodels.formula.api as smf
import numpy as np


class Graph:
    def __init__(self, merged_df, group: str):
        self.group = group
        self.merged_df = merged_df

    def partial_correlation(self):
        r_values = {}
        p_values = {}
        variables = ['Overall_Sleep_Score', 'Woke_Early_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']

        for var in variables:
            user_r_values = []
            user_p_values = []
            for user in self.merged_df['User'].unique():
                user_data = self.merged_df[self.merged_df['User'] == user]
                if (len(user_data)) > 2:
                    result = pg.partial_corr(data=user_data, x=var, y='Risk_Score', covar='Risk_Appeal_Score')
                    user_r_values.append(result['r'].values[0])
                    user_p_values.append(result['p-val'].values[0])

            r_values[var] = np.nanmean(user_r_values)
            p_values[var] = np.nanmean(user_p_values)

        corr_df = pd.DataFrame([r_values], index=['Risk_Score'])
        annotations = pd.DataFrame(index=['Risk_Score'], columns=r_values.keys())

        for col in r_values.keys():
            r_val = r_values[col]
            p_val = p_values[col]
            if p_val < 0.05:
                annotations.at['Risk_Score', col] = f'{r_val:.3f} $\mathbf{{(p={p_val:.3f}}}$)'
            else:
                annotations.at['Risk_Score', col] = f'{r_val:.3f} (p={p_val:.3f})'

        plt.figure(figsize=(14, 3))

        sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                    annot_kws={"size": 12})

        plt.title(f'Partial Correlations (r-values): {self.group}', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def pair_plot(self):
        sns.pairplot(self.merged_df[
                         ['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score',
                          'Risk Appeal Score', 'Risk Score']])
        plt.show()

    def risk_appeal_regression(self):

        # 1. Fit the mixed-effects model
        model = smf.mixedlm(
            "Risk_Score ~ Risk_Appeal_Score",  # Fixed effects formula
            self.merged_df,
            groups=self.merged_df["User"],  # Grouping variable for random effects
            re_formula="~Risk_Appeal_Score"  # Random slopes for Risk_Appeal_Score
        )
        result = model.fit()

        # 2. Extract overall slope and intercept
        global_intercept = result.params["Intercept"]
        global_slope = result.params["Risk_Appeal_Score"]
        p_value = result.pvalues["Risk_Appeal_Score"]

        plt.figure(figsize=(15, 8), dpi=350)

        # Create color palette for each user
        unique_users = self.merged_df["User"].unique()
        palette = plt.get_cmap("tab10")  # Use tab10 for distinct colors
        user_colors = {user: palette(i % 10) for i, user in enumerate(unique_users)}  # Cycle colors if > 10 users

        for user in unique_users:
            user_data = self.merged_df[self.merged_df["User"] == user]
            user_intercept = global_intercept + result.random_effects[user][0]  # Random intercept for this user
            user_slope = global_slope + result.random_effects[user][1]  # Random slope for this user

            # Plot user's data points
            sns.scatterplot(
                x="Risk_Appeal_Score",
                y="Risk_Score",
                data=user_data,
                color=user_colors[user],
                alpha=0.7,
                label=f"User {user}"  # Add labels for legend
            )

            # Plot user's regression line
            x_vals = np.linspace(user_data["Risk_Appeal_Score"].min(), user_data["Risk_Appeal_Score"].max(), 100)
            y_vals = user_intercept + user_slope * x_vals
            plt.plot(x_vals, y_vals, color=user_colors[user], alpha=0.7)

        # 5. Plot the overall line (fixed effect)
        x_vals = np.linspace(self.merged_df["Risk_Appeal_Score"].min(), self.merged_df["Risk_Appeal_Score"].max(), 100)
        y_vals_global = global_intercept + global_slope * x_vals
        plt.plot(x_vals, y_vals_global, color="black", linewidth=2, label="Overall Trend")

        plt.text(
            0.05, 0.95, f"-Log10 P-value: {-math.log(p_value, 10):.3f}",  # Adjust position and format of the text
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
        )

        # Final touches
        ax = plt.gca()
        ax.get_legend().remove()
        plt.title("Risk vs. Risk Appeal with Mixed Effects Model")
        plt.xlabel("Risk Appeal Score")
        plt.ylabel("Risk Score")
        plt.show()

    def correlation_between_sleep_and_risk(self):
        r_values = {}
        p_values = {}
        variables = ['Overall_Sleep_Score', 'Woke_Early_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']

        # Fit the mixed model for both sleep and risk scores
        for var in variables:
            # Fit the mixed-effects model for sleep score
            sleep_model = smf.mixedlm(f"{var} ~ Risk_Appeal_Score", self.merged_df, groups=self.merged_df["User"], re_formula="~Risk_Appeal_Score")
            sleep_result = sleep_model.fit()
            residuals_sleep = sleep_result.resid

            # Fit the mixed-effects model for risk score
            risk_model = smf.mixedlm("Risk_Score ~ Risk_Appeal_Score", self.merged_df, groups=self.merged_df["User"], re_formula="~Risk_Appeal_Score")
            risk_result = risk_model.fit()
            residuals_risk = risk_result.resid

            # Compute the correlation between the residuals
            corr, p_val = stats.pearsonr(residuals_sleep, residuals_risk)

            r_values[var] = corr
            p_values[var] = p_val

        # Output the correlations and p-values
        corr_df = pd.DataFrame([r_values], index=['Risk_Score'])
        annotations = pd.DataFrame(index=['Risk_Score'], columns=r_values.keys())

        for col in r_values.keys():
            r_val = r_values[col]
            p_val = p_values[col]
            if p_val < 0.05:
                annotations.at['Risk_Score', col] = f'{r_val:.3f} $\mathbf{{(p={p_val:.4f}}}$)'
            else:
                annotations.at['Risk_Score', col] = f'{r_val:.3f} (p={p_val:.3f})'

        plt.figure(figsize=(14, 3), dpi=350)
        sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                    annot_kws={"size": 12})
        plt.title(f'Partial Correlations (r-values): {self.group}', fontweight='bold')
        plt.tight_layout()
        plt.show()

