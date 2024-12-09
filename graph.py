import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
import pingouin as pg
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.stats.mediation import Mediation
import statsmodels.formula.api as smf
import numpy as np


class Graph:
    def __init__(self, merged_df, group: str):
        self.group = group
        self.merged_df = merged_df

    def show_risk_risk_appeal_regression(self):
        if len(self.merged_df) < 3:
            return
        r_value, p_value = pearsonr(self.merged_df['Risk_Appeal_Score'], self.merged_df['Risk_Score'])
        plt.figure(figsize=(10, 6))
        sns.regplot(x='Risk_Appeal_Score', y='Risk_Score', data=self.merged_df)
        plt.text(0.1, 0.9, f'R-value = {r_value:.2f} \nP-value: {p_value:.4}', transform=plt.gca().transAxes)
        plt.title(f'Regression of Risk vs. Risk Appeal Score, Group: {self.group}')
        plt.xlabel('Risk Appeal Score')
        plt.ylabel('Daily Risk Taken')
        plt.show()

    def partial_correlation(self):
        r_values = {}
        p_values = {}
        variables = ['Overall_Sleep_Score', 'Woke_Early_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']

        for var in variables:
            result = pg.partial_corr(data=self.merged_df, x=var, y='Risk_Score', covar='Risk_Appeal_Score')
            r_values[var] = result['r'].values[0]
            p_values[var] = result['p-val'].values[0]

        corr_df = pd.DataFrame([r_values], index=['Risk_Score'])
        annotations = pd.DataFrame(index=['Risk_Score'], columns=r_values.keys())

        for col in r_values.keys():
            r_val = r_values[col]
            p_val = p_values[col]
            if p_val < 0.05:
                annotations.at['Risk Score', col] = f'{r_val:.3f} $\mathbf{{(p={p_val:.3f}}}$)'
            else:
                annotations.at['Risk Score', col] = f'{r_val:.3f} (p={p_val:.3f})'

        plt.figure(figsize=(14, 3))

        sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                    annot_kws={"size": 12})

        plt.title(f'Partial Correlations (r-values): {self.group}', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def no_risk_correlation(self):
        r_values = {}
        p_values = {}
        variables = ['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score']

        for var in variables:
            result = pg.corr(self.merged_df[var], self.merged_df['Risk Score'])
            r_values[var] = result['r'].values[0]
            p_values[var] = result['p-val'].values[0]

        corr_df = pd.DataFrame([r_values], index=['Risk Score'])
        annotations = pd.DataFrame(index=['Risk Score'], columns=r_values.keys())

        for col in r_values.keys():
            r_val = r_values[col]
            p_val = p_values[col]
            if p_val < 0.05:
                annotations.at['Risk Score', col] = f'{r_val:.3f} $\mathbf{{(p={p_val:.3f}}}$)'
            else:
                annotations.at['Risk Score', col] = f'{r_val:.3f} (p={p_val:.3f})'

        plt.figure(figsize=(14, 3))

        sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                    annot_kws={"size": 12})

        plt.title(f'Sleep vs Risk Correlation (r-values): {self.group}', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def no_risk_correlation(self):
        r_values = {}
        p_values = {}
        variables = ['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score']

        for var in variables:
            result = pg.corr(self.merged_df[var], self.merged_df['Risk Score'])
            r_values[var] = result['r'].values[0]
            p_values[var] = result['p-val'].values[0]

        corr_df = pd.DataFrame([r_values], index=['Risk Score'])
        annotations = pd.DataFrame(index=['Risk Score'], columns=r_values.keys())

        for col in r_values.keys():
            r_val = r_values[col]
            p_val = p_values[col]
            if p_val < 0.05:
                annotations.at['Risk Score', col] = f'{r_val:.3f} $\mathbf{{(p={p_val:.3f}}}$)'
            else:
                annotations.at['Risk Score', col] = f'{r_val:.3f} (p={p_val:.3f})'

        plt.figure(figsize=(14, 3))

        sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                    annot_kws={"size": 12})

        plt.title(f'Sleep vs Risk Correlation (r-values): {self.group}', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def linear_regression(self):
        coef_values = {}
        p_values = {}
        variables = ['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score']

        for var in variables:
            X = self.merged_df[[var, 'Risk Appeal Score']]
            y = self.merged_df['Risk Score']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            coef_values[var] = model.params[var]
            p_values[var] = model.pvalues[var]

        coef_df = pd.DataFrame([coef_values], index=['Risk Score'])
        annotations = pd.DataFrame(index=['Risk Score'], columns=coef_values.keys())

        for col in coef_values.keys():
            coef_val = coef_values[col]
            p_val = p_values[col]
            if p_val < 0.05:
                annotations.at['Risk Score', col] = f'{coef_val:.3f} $\mathbf{{(p={p_val:.3f}}}$)'
            else:
                annotations.at['Risk Score', col] = f'{coef_val:.3f} (p={p_val:.3f})'

        plt.figure(figsize=(14, 3))
        sns.heatmap(coef_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                    annot_kws={"size": 12})

        plt.title(f'Linear Regression Coefficients: {self.group}', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def linear_regression_together(self):
        X = self.merged_df[['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score',
                            'Risk Appeal Score']]
        y = self.merged_df['Risk Score']  # Dependent variable

        X = sm.add_constant(X)  # Add constant (intercept) term
        model = sm.OLS(y, X).fit()  # Fit the linear regression model
        print(model.summary())  # Print summary of results

    def interation_term_analysis(self):
        coef_values = {}
        p_values = {}
        variables = ['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score']

        for var in variables:
            self.merged_df[f'{var}_x_RiskAppeal'] = self.merged_df[var] * self.merged_df['Risk Appeal Score']
            X = self.merged_df[[var, 'Risk Appeal Score', f'{var}_x_RiskAppeal']]
            y = self.merged_df['Risk Score']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            coef_values[var] = model.params[var]
            p_values[var] = model.pvalues[var]

        coef_df = pd.DataFrame([coef_values], index=['Risk Score'])
        annotations = pd.DataFrame(index=['Risk Score'], columns=coef_values.keys())

        for col in coef_values.keys():
            coef_val = coef_values[col]
            p_val = p_values[col]
            if p_val < 0.05:
                annotations.at['Risk Score', col] = f'$\mathbf{{{coef_val:.3f}}}$ (p=$\mathbf{{{p_val:.3f}}}$)'
            else:
                annotations.at['Risk Score', col] = f'{coef_val:.3f} (p={p_val:.3f})'

        plt.figure(figsize=(14, 3))
        sns.heatmap(coef_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                    annot_kws={"size": 12})

        plt.title(f'Integration Term Analysis Coefficients: {self.group}', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def random_forest(self):
        X = self.merged_df[['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score',
                            'Risk Appeal Score']]
        y = self.merged_df['Risk Score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100)
        rf_model.fit(X_train, y_train)

        predictions = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'Mean Squared Error: {mse}')

        # Feature importance plot
        plt.figure(figsize=(20, 20))
        feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
        feature_importance.sort_values().plot(kind='barh')
        plt.title('Feature Importance')
        plt.show()

    def pair_plot(self):
        sns.pairplot(self.merged_df[
                         ['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score',
                          'Risk Appeal Score', 'Risk Score']])
        plt.show()

    def mediation_analysis(self):
        for var in ['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score']:
            # Independent variable: Sleep score
            X = self.merged_df[var]
            # Mediator: Risk Appeal Score
            M = self.merged_df['Risk Appeal Score']
            # Dependent variable: Risk Score
            Y = self.merged_df['Risk Score']

            # Fit the mediator model (M ~ X)
            mediator_model = sm.OLS(M, sm.add_constant(X)).fit()

            # Fit the outcome model (Y ~ X + M)
            outcome_model = sm.OLS(Y, sm.add_constant(pd.concat([X, M], axis=1))).fit()

            # Perform mediation analysis
            med = Mediation(outcome_model, mediator_model, exposure=var, mediator='Risk Appeal Score')
            med_summary = med.fit()

            # Print summary of mediation analysis
            print(f"Mediation Analysis for {var}:")
            print(med_summary.summary())

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

    def mixed_model_regression(self):
        if len(self.merged_df) < 3:
            return

        # Fit a mixed-effects model
        model = smf.mixedlm("Risk_Score ~ Risk_Appeal_Score", self.merged_df, groups=self.merged_df['User'])
        result = model.fit()

        # Extract slope (coefficient) and p-value for Risk Appeal Score
        slope = result.params['Risk_Appeal_Score']
        p_value = result.pvalues['Risk_Appeal_Score']

        # Plot the data points
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Risk_Appeal_Score', y='Risk_Score', data=self.merged_df, hue='User', palette='viridis',
                        alpha=0.7, legend=False)
        x_vals = np.linspace(self.merged_df['Risk_Appeal_Score'].min(), self.merged_df['Risk_Appeal_Score'].max(), 100)
        y_vals = result.params['Intercept'] + slope * x_vals
        plt.plot(x_vals, y_vals, color='red', label=f'Regression Line')

        plt.text(0.05, 0.9, f'Slope = {slope:.2f}\nP-value: {p_value:.4f}', transform=plt.gca().transAxes)

        plt.title(f'Regression of Risk vs. Risk Appeal Score, Group: {self.group}')
        plt.xlabel('Risk Appeal Score')
        plt.ylabel('Daily Risk Taken')

        plt.show()

    def mixed_model_partial_correlation(self):
        r_values = {}
        p_values = {}
        variables = ['Overall_Sleep_Score', 'Woke_Early_Score', 'Woke_Many_Times_Score', 'Sleep_Latency_Score']

        # Residualize Risk_Score for Risk_Appeal_Score
        base_model = sm.OLS(self.merged_df['Risk_Score'], sm.add_constant(self.merged_df['Risk_Appeal_Score'])).fit()
        self.merged_df['Residual_Risk_Score'] = base_model.resid

        for var in variables:
            # Residualize the predictor (var) for Risk_Appeal_Score
            var_model = sm.OLS(self.merged_df[var], sm.add_constant(self.merged_df['Risk_Appeal_Score'])).fit()
            self.merged_df[f'Residual_{var}'] = var_model.resid

            # Fit mixed model with residuals
            model = smf.mixedlm(f"Risk_Score ~ {var}", self.merged_df, groups=self.merged_df['User'])
            result = model.fit()

            # Predict and calculate correlation between predicted and actual residuals
            r_values[var] = result.params[f'{var}']
            p_values[var] = result.pvalues[f'{var}']

        # Create DataFrame for heatmap
        corr_df = pd.DataFrame([r_values], index=['Risk_Score'])
        annotations = pd.DataFrame(index=['Risk_Score'], columns=r_values.keys())

        for col in r_values.keys():
            r_val = r_values[col]
            p_val = p_values[col]
            if p_val < 0.05:
                annotations.at['Risk_Score', col] = f'{r_val:.3f} $\mathbf{{(p={p_val:.3f})}}$'
            else:

                annotations.at['Risk_Score', col] = f'{r_val:.3f} (p={p_val:.3f})'

        # Plot heatmap
        plt.figure(figsize=(14, 3))
        sns.heatmap(corr_df, annot=annotations.values, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='',
                    annot_kws={"size": 12})

        plt.title(f'Partial Correlations with Mixed Model: {self.group}', fontweight='bold')
        plt.tight_layout()
        plt.show()
