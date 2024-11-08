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

    def partial_correlation(self):
        r_values = {}
        p_values = {}
        variables = ['Overall Sleep Score', 'Woke Early Score', 'Woke Many Times Score', 'Sleep Latency Score']

        for var in variables:
            result = pg.partial_corr(data=self.merged_df, x=var, y='Risk Score', covar='Risk Appeal Score')
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

        plt.title(f'Partial Correlations (r-values): {self.group}', fontweight='bold')

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
