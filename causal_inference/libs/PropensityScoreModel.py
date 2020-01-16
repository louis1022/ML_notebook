import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class PropensityScoreModel():

    def __init__(self, raw_df, treatment, covariate, outcomes, model, params, scaler):
        self.raw_df = raw_df
        self.treatment = treatment
        self.covariate = covariate
        self.outcomes = outcomes

        self.model, self.scores = self._fit(
            raw_df[covariate], raw_df[treatment], model, params, scaler)
        # self.ate = self.estimate_ate(raw_df, self.scores, outcomes)

    def _fit(self, X, y, model=None, params=None, scaler=None):
        if not model:
            model = LogisticRegression
            params = {'solver': 'lbfgs', 'max_iter': 200, 'random_state': self.random_state}
            scaler = preprocessing.MinMaxScaler()

        if scaler:
            X = pd.DataFrame(scaler.fit_transform(X))

        clf = model(**params)
        clf.fit(X, y)
        pred_proba = clf.predict_proba(X)[:, 1]
        return clf, pred_proba

    def _estimate_effect(self, weight, threshhold=(0.02, 0.98)):
        e_y1 = []
        e_y0 = []
        effect_size = []

        is_use_data = (self.scores > threshhold[0]) & (self.scores < threshhold[1])
        for outcome in self.outcomes:
            X = pd.get_dummies(self.raw_df[self.treatment][is_use_data])
            y = self.raw_df[outcome][is_use_data]
            clf = LinearRegression(fit_intercept=False)
            clf.fit(X, y, sample_weight=weight[is_use_data])

            y1 = clf.predict([[0, 1]])[0]
            y0 = clf.predict([[1, 0]])[0]

            e_y1.append(y1)  # 処置群であるときの予測値
            e_y0.append(y0)  # 非処置群であるときの予測値
            effect_size.append(y1 - y0)
        return [e_y1, e_y0, effect_size]

    def estimate_ate(self):
        ate_weight = np.where(
            self.raw_df[self.treatment] == 1, 1 / self.scores, 1 / (1 - self.scores))
        effect_df = pd.DataFrame(
            self._estimate_effect(ate_weight),
            columns=self.outcomes,
            index=['Treated', 'Untreated', 'ATE']).T
        return effect_df

    def plot_ajusted_effect(self, effect_df):
        for index, row in effect_df.iterrows():
            plt.title(index)
            plt.ylabel('Effect Size')
            row[['Untreated', 'Treated']].plot.bar(
                color='steelblue', rot=0, align="center", label=f"ATE = {row['ATE'].round(2)}")
            plt.legend()
            plt.show()

    def plot_roc_curve(self, figsize=None):
        y_actual = self.raw_df[self.treatment]
        y_proba = self.scores

        fpr, tpr, _ = roc_curve(y_actual, y_proba)
        auc = roc_auc_score(y_actual, y_proba)

        plt.figure(figsize=None)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def plot_layer_effect(self, n_layer=3, figsize=None):
        df = self.raw_df
        treatment = self.treatment
        scores = self.scores

        df[treatment].replace({0: 'Untreated', 1: 'Treated'}, inplace=True)

        scores_rank = pd.qcut(scores, n_layer, labels=False)
        for outcome in self.outcomes:
            plt.figure(figsize=(25, 5 * ((n_layer - 1) // 5 + 1)))
            for i in range(n_layer):
                plt.subplot((n_layer - 1) // 5 + 1, 5, i + 1)
                sns.barplot(
                    x=df.loc[scores_rank == i, treatment],
                    y=df.loc[(scores_rank == i) & (df[outcome] > 0), outcome]
                ).set_title(f'{i+1} Layer')
            plt.tight_layout()
            plt.show()

    def ajusted_covariate(self, is_weited=True):
        df = self.raw_df[self.covariate + [self.treatment]].copy()
        treatment = self.treatment
        df['weight'] = self.scores if is_weited else np.ones(df.shape[0])

        dst_df = pd.DataFrame()
        for is_treat, grouped_df in df.groupby(treatment):
            group_name = 'Treated' if is_treat == 1 else 'Untreated'

            avg = np.average(grouped_df, weights=grouped_df['weight'], axis=0)
            var = np.average((grouped_df - avg)**2, weights=grouped_df['weight'], axis=0)
            std = np.sqrt(var)
            stderr = std / (np.sqrt(grouped_df['weight'].sum() - 1))

            dst_df[f'{group_name}_mean'] = avg
            dst_df[f'{group_name}_var'] = var

        dst_df['d'] = (dst_df['Treated_mean'] - dst_df['Untreated_mean']
                       ) / np.sqrt((dst_df['Treated_var'] + dst_df['Untreated_var']) / 2)
        return dst_df
