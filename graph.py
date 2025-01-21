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
    def __init__(self, merged_df, group: str, groups_dfs):
        self.group = group
        self.groups_dfs = groups_dfs
        self.merged_df = merged_df

    def risk_expected_value_regression(self):
        self.plot_regression('Expected_Value', 'Risk_Score')

    def correlation_between_sleep_and_risk(self):
        fig = plt.figure(figsize=(14, len(self.groups_dfs) * 3.4), dpi=350)
        for i, group in enumerate(self.groups_dfs, start=1):
            ax = fig.add_subplot(len(self.groups_dfs), 1, i)
            dataframe = self.groups_dfs[group]
            r_values = {}
            p_values = {}
            variables = ['Overall_Sleep_Score', 'Woke_Early_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']

            # Fit the mixed model for both sleep and risk scores
            for var in variables:
                # Fit the mixed-effects model for sleep score
                sleep_model = smf.mixedlm(f"{var} ~ Expected_Value", dataframe, groups=dataframe["User"])
                sleep_result = sleep_model.fit()
                residuals_sleep = sleep_result.resid

                # Fit the mixed-effects model for risk score
                risk_model = smf.mixedlm("Risk_Score ~ Expected_Value", dataframe, groups=dataframe["User"])
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

            sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                        annot_kws={"size": 12}, ax=ax)
            ax.set_title(f"{group}", fontsize=12, fontweight='bold')

        fig.suptitle(f'Sleep/Risk\nPartial Correlations (r-values)', fontweight='bold')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.5)
        plt.show()

    def correlation_between_mood_and_risk(self):
        fig = plt.figure(figsize=(20, len(self.groups_dfs) * 3.4), dpi=350)
        for i, group in enumerate(self.groups_dfs, start=1):
            ax = fig.add_subplot(len(self.groups_dfs), 1, i)
            dataframe = self.groups_dfs[group]
            r_values = {}
            p_values = {}
            variables = ['Valence', 'Arousal', 'Anxious', 'Elated', 'Sad', 'Irritable', 'Energetic']

            for var in variables:
                mood_residual = smf.mixedlm(f"{var} ~ Expected_Value", dataframe, groups=dataframe["User"]).fit().resid
                residuals_risk = smf.mixedlm("Risk_Score ~ Expected_Value", dataframe,
                                             groups=dataframe["User"]).fit().resid
                corr, p_val = stats.pearsonr(mood_residual, residuals_risk)

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

            sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                        annot_kws={"size": 12}, ax=ax)
            ax.set_title(f"{group}", fontsize=12, fontweight='bold')

        fig.suptitle(f'Mood/Risk\nPartial Correlations (r-values)', fontweight='bold')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.5)
        plt.show()

    def correlation_between_sleep_mood(self):
        mood_variables = ['Valence', 'Arousal', 'Anxious', 'Elated', 'Sad', 'Irritable', 'Energetic']
        sleep_variables = ['Overall_Sleep_Score', 'Woke_Early_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']

        fig = plt.figure(figsize=(20, len(self.groups_dfs) * 3.4), dpi=350)
        for i, group in enumerate(self.groups_dfs, start=1):
            ax = fig.add_subplot(len(self.groups_dfs), 1, i)
            dataframe = self.groups_dfs[group]
            r_values = {mood: {} for mood in mood_variables}
            p_values = {mood: {} for mood in mood_variables}

            for mood in mood_variables:
                for sleep in sleep_variables:
                    model = smf.mixedlm(f"{mood} ~ {sleep}", dataframe, groups=dataframe["User"]).fit()
                    corr, p_val = model.params[f"{sleep}"], model.pvalues[f"{sleep}"]
                    r_values[mood][sleep] = corr
                    p_values[mood][sleep] = p_val

            corr_df = pd.DataFrame(r_values).T
            annotations = pd.DataFrame(index=mood_variables, columns=sleep_variables)

            for mood in mood_variables:
                for sleep in sleep_variables:
                    r_val = r_values[mood][sleep]
                    p_val = p_values[mood][sleep]
                    if p_val < 0.05:
                        annotations.at[mood, sleep] = f'{r_val:.3f} $\mathbf{{(p={p_val:.4f})}}$'
                    else:
                        annotations.at[mood, sleep] = f'{r_val:.3f} (p={p_val:.3f})'

            sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                        annot_kws={"size": 12}, ax=ax)
            ax.set_title(f"Group: {group}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Sleep Variables", fontsize=10)
            ax.set_ylabel("Mood Variables", fontsize=10)

        fig.suptitle(f'Mood/Sleep\nPartial Correlations (r-values)', fontweight='bold', fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.5)
        plt.show()

    def sleep_risk_regression(self):
        sleep_variables = ['Overall_Sleep_Score', 'Woke_Early_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']
        for sleep_variables in sleep_variables:
            self.plot_regression('Risk_Score', sleep_variables)

    def plot_regression(self, var1, var2):
        fig = plt.figure(figsize=(14, len(self.groups_dfs) * 4), dpi=350)
        for i, group in enumerate(self.groups_dfs, start=1):
            ax = fig.add_subplot(len(self.groups_dfs), 1, i)
            dataframe = self.groups_dfs[group]
            model = smf.mixedlm(
                f"{var1} ~ {var2}",
                dataframe,
                groups=dataframe["User"],
                re_formula=f"~{var2}"
            )
            result = model.fit()
            global_intercept = result.params["Intercept"]
            global_slope = result.params[f"{var2}"]
            p_value = result.pvalues[f"{var2}"]
            unique_users = dataframe["User"].unique()
            colors = distinctipy.get_colors(len(unique_users))
            user_colors = {user: color for user, color in zip(unique_users, colors)}
            for user in unique_users:
                user_data = dataframe[dataframe["User"] == user]
                user_intercept = global_intercept + result.random_effects[user][0]
                user_slope = global_slope + result.random_effects[user][1]
                sns.scatterplot(
                    x=f"{var2}",
                    y=f"{var1}",
                    data=user_data,
                    color=user_colors[user],
                    alpha=0.7,
                    label=f"User {user}",
                    ax=ax,
                    legend=False
                )
                x_vals = np.linspace(user_data[f"{var2}"].min(), user_data[f"{var2}"].max(), 100)
                y_vals = user_intercept + user_slope * x_vals
                ax.plot(x_vals, y_vals, color=user_colors[user], alpha=0.7)

            x_vals = np.linspace(dataframe[f"{var2}"].min(), dataframe[f"{var2}"].max(), 100)
            y_vals_global = global_intercept + global_slope * x_vals
            ax.plot(x_vals, y_vals_global, color="black", linewidth=2, label="Overall Trend")

            ax.text(
                0.05, 0.95, f"P-value: {p_value:.3f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
            )

            ax.set_title(f"{group}", fontsize=12, fontweight='bold')
            ax.set_xlabel(f"{var2}")
            ax.set_ylabel(f"{var1}", )

        plt.subplots_adjust(hspace=0.3)
        fig.suptitle(f"{var1}/{var2}\nMixed Effects Model on Users", fontweight='bold')
        plt.show()

    def mediation_analysis_from_scratch(self):
        dataframe = self.groups_dfs["Both Groups"]
        sleep_variables = ['Overall_Sleep_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']
        mood_variables = ['Valence', 'Arousal', 'Anxious', 'Elated']

        # Normalize the Risk Score and Expected Value between 0-100.
        dataframe['Risk_Score'] = 100 * (dataframe['Risk_Score'] - dataframe['Risk_Score'].min()) / (
                dataframe['Risk_Score'].max() - dataframe['Risk_Score'].min())
        dataframe['Expected_Value'] = 100 * (dataframe['Expected_Value'] - dataframe['Expected_Value'].min()) / (
                dataframe['Expected_Value'].max() - dataframe['Expected_Value'].min())

        # Get the residuals for Risk Score by removing the Expected Value.
        residuals_risk = smf.mixedlm("Risk_Score ~ Expected_Value", dataframe, groups=dataframe["User"]).fit().resid
        dataframe["Risk_Score"] = residuals_risk
        for sleep in sleep_variables:
            dataframe[f"{sleep}"] = smf.mixedlm(f"{sleep} ~ Expected_Value", dataframe,
                                                groups=dataframe["User"]).fit().resid
            for mood in mood_variables:
                dataframe[f"{mood}"] = smf.mixedlm(f"{mood} ~ Expected_Value", dataframe,
                                                   groups=dataframe["User"]).fit().resid

                # Step1 - Correlation Sleep and Risk:
                model = smf.mixedlm(f"Risk_Score ~ {sleep}", dataframe, groups=dataframe["User"]).fit()
                sleep_risk_p_value = model.pvalues[f"{sleep}"]

                # Step2 - Correlation Sleep to Mood:
                model = smf.mixedlm(f"{mood} ~ {sleep} ", dataframe, groups=dataframe["User"]).fit()
                sleep_mood_p_value = model.pvalues[f"{sleep}"]

                should_continue = False
                if sleep_risk_p_value <= 0.05 and sleep_mood_p_value <= 0.05:
                    should_continue = True

                # Step3 - Correlation Sleep + Mood → Risk (X + M → Y)
                if should_continue:
                    mediation = pg.mediation_analysis(data=dataframe, x=sleep, m=mood, y='Risk_Score')
                    a_coeff = mediation.loc[mediation['path'] == f"{mood} ~ X", "coef"].values[0]
                    b_coeff = mediation.loc[mediation['path'] == f"Y ~ {mood}", "coef"].values[0]
                    total_coeff = mediation.loc[mediation['path'] == "Total", "coef"].values[0]
                    indirect_coeff = mediation.loc[mediation['path'] == "Indirect", "coef"].values[0]
                    a_p = mediation.loc[mediation['path'] == f"{mood} ~ X", "pval"].values[0]
                    b_p = mediation.loc[mediation['path'] == f"Y ~ {mood}", "pval"].values[0]
                    total_p = mediation.loc[mediation['path'] == "Total", "pval"].values[0]
                    indirect_p = mediation.loc[mediation['path'] == "Indirect", "pval"].values[0]
                    plot_mediation_graph(sleep, mood, a_coeff, b_coeff, total_coeff, indirect_coeff, a_p, b_p, total_p,
                                         indirect_p)


def plot_mediation_graph(sleep, mood, a_coeff, b_coeff, total_coeff, indirect_coeff, a_p, b_p, total_p, indirect_p):
    G = nx.DiGraph()

    G.add_node(f"{sleep} (X)")
    G.add_node(f"{mood} (M)")
    G.add_node(f"Risk (Y)")

    G.add_edge(f"{sleep} (X)", f"{mood} (M)", label=f"{a_coeff:.3f} (p={a_p:.3f})")
    G.add_edge(f"{mood} (M)", "Risk (Y)", label=f"{b_coeff:.3f} (p={b_p:.3f})")
    G.add_edge(f"{sleep} (X)", "Risk (Y)",
               label=f"Total: {total_coeff:.3f} (p={total_p:.3f}) | Indirect: {indirect_coeff:.3f} (p={indirect_p:.3f})")

    pos = {f"{sleep} (X)": (0, 0), f"{mood} (M)": (1, 1), "Risk (Y)": (2, 0)}

    plt.figure(figsize=(13, 8))
    ax = plt.gca()

    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=40, edge_color="black")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)

    color_map = ['#77dd77', '#cdb4db', '#ffcccb']
    for i, (node, (x, y)) in enumerate(pos.items()):
        ax.text(
            x, y, node, fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=2.1", edgecolor="black", facecolor=color_map[i % len(color_map)])
        )

    plt.title(f"Mediation Analysis Between: {sleep} (X), {mood} (M), and Risk (Y)", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.show()
