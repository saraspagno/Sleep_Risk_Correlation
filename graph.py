import math
from statistics import mean

import distinctipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import stats
import pingouin as pg
import statsmodels.formula.api as smf
import numpy as np
import networkx as nx


class Graph:
    def __init__(self, merged_df, group: str):
        self.group = group
        self.merged_df = merged_df

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
        colors = distinctipy.get_colors(len(unique_users))
        user_colors = {user: color for user, color in zip(unique_users, colors)}

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
        plt.title(f"Risk vs. Risk Appeal with Mixed Effects Model, {self.group}")
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
            sleep_model = smf.mixedlm(f"{var} ~ Risk_Appeal_Score", self.merged_df, groups=self.merged_df["User"],
                                      re_formula="~Risk_Appeal_Score")
            sleep_result = sleep_model.fit()
            residuals_sleep = sleep_result.resid

            # Fit the mixed-effects model for risk score
            risk_model = smf.mixedlm("Risk_Score ~ Risk_Appeal_Score", self.merged_df, groups=self.merged_df["User"],
                                     re_formula="~Risk_Appeal_Score")
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

    def mediation_analysis(self):
        for sleep in ['Overall_Sleep_Score', 'Woke_Early_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']:
            for mood in ['Arousal', 'Anxious', 'Elated', 'Sad', 'Irritable', 'Energetic']:
                sleep_model = smf.mixedlm(f"{sleep} ~ Risk_Appeal_Score", self.merged_df, groups=self.merged_df["User"],
                                          re_formula="~Risk_Appeal_Score")
                sleep_result = sleep_model.fit()
                residuals_sleep = sleep_result.resid
                self.merged_df[sleep] = residuals_sleep

                risk_model = smf.mixedlm("Risk_Score ~ Risk_Appeal_Score", self.merged_df,
                                         groups=self.merged_df["User"],
                                         re_formula="~Risk_Appeal_Score")
                risk_result = risk_model.fit()
                self.merged_df['Risk_Score'] = risk_result.resid

                mediation = pg.mediation_analysis(data=self.merged_df, x=sleep, m=mood, y='Risk_Score')
                a = mediation.loc[mediation['path'] == f"{mood} ~ X", "coef"].values[0]
                b = mediation.loc[mediation['path'] == f"Y ~ {mood}", "coef"].values[0]
                c = mediation.loc[mediation['path'] == "Direct", "coef"].values[0]
                total = mediation.loc[mediation['path'] == "Total", "coef"].values[0]
                if abs(c) > 0.01 or abs(total) > 0.01:
                    plot_mediation_graph(sleep, mood, a, b, c, total)


def plot_mediation_graph(sleep, mood, a, b, c, total):
    G = nx.DiGraph()

    G.add_node(f"{sleep} (X)")
    G.add_node(f"{mood} (M)")
    G.add_node(f"Risk (Y)")

    # Add edges with coefficients
    G.add_edge(f"{sleep} (X)", f"{mood} (M)", label=f"{a:.2f}")
    G.add_edge(f"{mood} (M)", "Risk (Y)", label=f"{b:.2f} ")
    G.add_edge(f"{sleep} (X)", "Risk (Y)", label=f"Direct: {c:.2f} | Total: {total:.2f}")

    pos = {f"{sleep} (X)": (0, 0), f"{mood} (M)": (1, 1), "Risk (Y)": (2, 0)}

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=40, edge_color="black")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    color_map = ['#77dd77', '#cdb4db', '#ffcccb']
    for i, (node, (x, y)) in enumerate(pos.items()):
        ax.text(
            x, y, node, fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=color_map[i % len(color_map)])
        )

    plt.title(f"Mediation Analysis Between {sleep} (X), {mood} (M), and Risk (Y)", fontsize=14)
    plt.axis("off")
    plt.show()
