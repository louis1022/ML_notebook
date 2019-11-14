import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


class PropensityScoreMatching():

    def __init__(self, raw_df, treatment):
        self.raw_df = raw_df
        self.treatment = treatment

    def fit(self, model=None, params=None, scaler=None, n_fold=2):
        if not model:
            model = LogisticRegression
            params = {'solver': 'lbfgs', 'max_iter': 200, 'random_state': self.random_state}
            scaler = preprocessing.MinMaxScaler()

        df = self.raw_df.copy()
        p_score = np.zeros(df.shape[0])

        X = df.drop(self.treatment, axis=1)
        y = df[self.treatment]

        if scaler:
            X = pd.DataFrame(scaler.fit_transform(X))

        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=None)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = model(**params)
            clf.fit(X_train, y_train)
            p_score[test_index] = clf.predict_proba(X_test)[:, 1]

        self.propensity_score = p_score

    def predict(self):
        df = self.raw_df.copy()
        df['propensity_score'] = self.propensity_score

        one_df = df[df[self.treatment] == 1]
        zero_df = df[df[self.treatment] == 0]

        matched_idx = self._get_matched_idx(
            one_df['propensity_score'].values, zero_df['propensity_score'].values)

        transformed_df = pd.concat([
            one_df.iloc[matched_idx.flatten()], zero_df], axis=0).reset_index(drop=True)

        self.transformed_df = transformed_df
        self.atu = transformed_df.loc[transformed_df[self.treatment] == 0].mean(axis=0)
        self.att = transformed_df.loc[transformed_df[self.treatment] == 1].mean(axis=0)
        return transformed_df

    def _get_matched_idx(self, one_scores, zero_scores):
        neigh = NearestNeighbors(n_neighbors=1, metric='manhattan')
        neigh.fit(one_scores.reshape(-1, 1))
        distances, indices = neigh.kneighbors(zero_scores.reshape(-1, 1))
        return indices

    def estimate_ate(self, outcome):
        return self.att[outcome] - self.atu[outcome]

    def estimate_sem(self, outcome):
        df = self.transformed_df
        sem_zero = df.loc[df[self.treatment] == 0, outcome].sem()
        sem_one = df.loc[df[self.treatment] == 1, outcome].sem()
        return sem_zero, sem_one
