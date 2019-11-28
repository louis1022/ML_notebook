import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


class PropensityScoreMatching():

    def __init__(self, raw_df, treatment):
        self.raw_df = raw_df
        self.df = raw_df.copy()
        self.treatment = treatment

    def fit(self, model=None, params=None, scaler=None):
        if not model:
            model = LogisticRegression
            params = {'solver': 'lbfgs', 'max_iter': 200, 'random_state': self.random_state}
            scaler = preprocessing.MinMaxScaler()

        X = self.df.drop(self.treatment, axis=1)
        y = self.df[self.treatment]

        if scaler:
            X = pd.DataFrame(scaler.fit_transform(X))

        clf = model(**params)
        clf.fit(X, y)
        self.df['p_score'] = clf.predict_proba(X)[:, 1]

    def predict(self, threshold=0.005):
        df = self.df.copy()

        one_df = df[df[self.treatment] == 1]
        zero_df = df[df[self.treatment] == 0]

        distances, matched_idx = self._get_matched_idx(
            one_df['p_score'].values, zero_df['p_score'].values)

        if threshold is not None:
            matched_idx = matched_idx[distances < threshold]

        transformed_df = pd.concat(
            [one_df.iloc[matched_idx.flatten()], zero_df], axis=0).reset_index(drop=True)

        self.transformed_df = transformed_df
        return transformed_df

    def _get_matched_idx(self, one_scores, zero_scores):
        neigh = NearestNeighbors(n_neighbors=1, metric='manhattan')
        neigh.fit(one_scores.reshape(-1, 1))
        distances, indices = neigh.kneighbors(zero_scores.reshape(-1, 1))
        return distances, indices

    def estimate_ate(self, outcome, method='default'):
        if method == 'default':
            df = self.transformed_df
            e_y1 = df.loc[df[self.treatment] == 1, outcome].mean(axis=0)
            e_y0 = df.loc[df[self.treatment] == 0, outcome].mean(axis=0)
        elif method == 'ipw':
            z1 = df[self.treatment]
            ps = df['propensity_score']
            e_y1 = df[outcome].apply(lambda x: sum((z1 * x) / ps) / sum(z1 / ps), axis=0)
            e_y0 = df[outcome].apply(lambda x: sum(((1 - z1) * x) / (1 - ps)) / sum((1 - z1) / (1 - ps)), axis=0)
        return e_y1 - e_y0
