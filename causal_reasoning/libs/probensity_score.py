import numpy as np
import pandas as pd


from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


class PropensityScoreMatching():

    def __init__(self, raw_df, intervention, random_state=None, calibrate_propensity=True):
        self.raw_df = raw_df
        self.intervention = intervention
        self.random_state = random_state
        self.calibrate_propensity = calibrate_propensity
        self.propensity_score = None
        self.transformed_df = None

    def get_matched_idx(self):
        p_score = self.propensity_score
        intervention = self.raw_df[self.intervention]

        neigh = NearestNeighbors(n_neighbors=1, metric='manhattan')
        neigh.fit(p_score[intervention == 1].reshape(-1, 1))

        distances, indices = neigh.kneighbors(p_score[intervention == 0].reshape(-1, 1))
        return indices

    def transform(self):
        df = self.raw_df
        df['propensity_score'] = self.propensity_score
        matched_idx = self.get_matched_idx()
        transformed_df = pd.concat([
            df[df[self.intervention] == 1].iloc[matched_idx.flatten()],
            df[df[self.intervention] == 0]
        ], axis=0).reset_index(drop=True)
        self.transformed_df = transformed_df
        return transformed_df

    def fit(self, clf=None, scaler=None, n_fold=2):
        if not clf:
            clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=self.random_state)
        if not scaler:
            scaler = preprocessing.MinMaxScaler()

        df = self.raw_df.copy()
        p_score = np.zeros(df.shape[0])

        X = df.drop(self.intervention, axis=1)
        y = df[self.intervention]

        scaler.fit(X)
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=None)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=None)
            clf.fit(scaler.transform(X_train), y_train)
            p_score[test_index] = clf.predict_proba(scaler.transform(X_test))[:, 1]

        self.propensity_score = p_score

    def estimate_atu(self, outcome):
        df = self.transformed_df
        return df.loc[df[self.intervention] == 0, outcome].mean()

    def estimate_att(self, outcome):
        df = self.transformed_df
        return df.loc[df[self.intervention] == 1, outcome].mean()

    def estimate_ate(self, outcome):
        att = self.estimate_att(outcome)
        atu = self.estimate_atu(outcome)
        return att - atu

    def estimate_sem(self, outcome):
        df = self.transformed_df
        sem_zero = df.loc[df[self.intervention] == 0, outcome].mean()
        sem_one = df.loc[df[self.intervention] == 1, outcome].mean()
        return sem_zero, sem_one
