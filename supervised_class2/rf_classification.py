# https://deeplearningcourses.com/c/machine-learning-in-python-random-forest-adaboost
# https://www.udemy.com/machine-learning-in-python-random-forest-adaboost
# mushroom data from:
# https://archive.ics.uci.edu/ml/datasets/Mushroom
# put all files in the folder ../large_files/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

NUMERICAL_COLS = ()
CATEGORICAL_COLS = np.arange(22) + 1 # 1..22 inclusive

# transforms data from dataframe to numerical matrix
# one-hot encodes categories and normalizes numerical columns
# we want to use the scales found in training when transforming the test set
# so only call fit() once
# call transform() for any subsequent data
class DataTransformer:
  def fit(self, df):
    self.labelEncoders = {}
    self.scalers = {}
    for col in NUMERICAL_COLS:
      scaler = StandardScaler()
      scaler.fit(df[col].reshape(-1, 1))
      self.scalers[col] = scaler

    for col in CATEGORICAL_COLS:
      encoder = LabelEncoder()
      # in case the train set does not have 'missing' value but test set does
      values = df[col].tolist()
      values.append('missing')
      encoder.fit(values)
      self.labelEncoders[col] = encoder

    # find dimensionality
    self.D = len(NUMERICAL_COLS)
    for col, encoder in self.labelEncoders.iteritems():
      self.D += len(encoder.classes_)
    print "dimensionality:", self.D

  def transform(self, df):
    N, _ = df.shape
    X = np.zeros((N, self.D))
    i = 0
    for col, scaler in self.scalers.iteritems():
      X[:,i] = scaler.transform(df[col].as_matrix().reshape(-1, 1)).flatten()
      i += 1

    for col, encoder in self.labelEncoders.iteritems():
      # print "transforming col:", col
      K = len(encoder.classes_)
      X[np.arange(N), encoder.transform(df[col]) + i] = 1
      i += K
    return X

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)


def replace_missing(df):
  # standard method of replacement for numerical columns is median
  for col in NUMERICAL_COLS:
    if np.any(df[col].isnull()):
      med = np.median(df[ col ][ df[col].notnull() ])
      df.loc[ df[col].isnull(), col ] = med

  # set a special value = 'missing'
  for col in CATEGORICAL_COLS:
    if np.any(df[col].isnull()):
      print col
      df.loc[ df[col].isnull(), col ] = 'missing'


def get_data():
  df = pd.read_csv('../large_files/mushroom.data', header=None)

  # replace label column: e/p --> 0/1
  # e = edible = 0, p = poisonous = 1
  df[0] = df.apply(lambda row: 0 if row[0] == 'e' else 1, axis=1)

  # check if there is missing data
  replace_missing(df)

  # transform the data
  transformer = DataTransformer()

  X = transformer.fit_transform(df)
  Y = df[0].as_matrix()
  return X, Y


if __name__ == '__main__':
  X, Y = get_data()

  # do a quick baseline test
  baseline = LogisticRegression()
  print "CV baseline:", cross_val_score(baseline, X, Y, cv=8).mean()

  # single tree
  tree = DecisionTreeClassifier()
  print "CV one tree:", cross_val_score(tree, X, Y, cv=8).mean()

  model = RandomForestClassifier(n_estimators=20) # try 10, 20, 50, 100, 200
  print "CV forest:", cross_val_score(model, X, Y, cv=8).mean()
