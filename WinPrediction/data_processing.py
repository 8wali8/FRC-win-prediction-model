import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import statistics
import sklearn.datasets
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
import sklearn.svm
import xgboost as xgb
from xgboost import XGBClassifier

data_path = "WinPrediction/2023 FRC World Championship Public Pre-Scouting Database - Archimedes Division.csv"
df = pd.read_csv(data_path, skiprows=2)
df = df.drop("Team Name\n(linked to Statbotics)", axis=1)

#remove this after
df = df.drop(["Drive Train Type", 'Game Piece Capability', 'Game Piece Acquisition', 'Game Piece Scoring', 'Center Autonomous Routine', 'Non-Cable Protector Autonomous Routine', 'Cable Protector Autonomous Routine', 'Endgame Assistance', 'Comments'], axis=1)
import numpy as numpy
from sklearn.model_selection import train_test_split

X_train, X_test= train_test_split(df ,shuffle = False)

def my_classifier(x, y, classifier, n_folds):
  listvar = []
  clf = SVC(kernel = 'linear', C = 1, random_state = 42)

  if classifier == "SVM":
    clf = SVC(kernel = 'linear', C = 1, random_state = 42)
    parameters = {"C": np.arange(1,2,1), "kernel": ['linear','sigmoid','precomputed']}
  elif classifier == "random tree":
    clf = RandomForestClassifier()
    parameters = {"max_depth":[30], "max_features": [3]}
  elif classifier == "xgboost":
    clf = XGBClassifier(random_state = 42)
    parameters = {"booster":["gbtree","gblinear"], "eta":[0,0.3,0.6,0.9]}

  kf = KFold(n_folds, shuffle = True)
  for train_index, test_index in kf.split(x):
      x_train, x_test = x.iloc[train_index], x.iloc[test_index]
      y_train, y_test = y[train_index], y[test_index]

      clf.fit(x_train, y_train)

      predictions = clf.predict(x_test)

      score = f1_score(y_test, predictions)
      listvar.append(score)
      print("F1 score: ", score)
    #   searcher = RandomizedSearchCV(clf, parameters)
    #   searcher.fit(x_train, y_train)
    #   print("Best params: ", searcher.best_params_)
    #   print("Best score: ", searcher.best_score_)
    #   y_pred = searcher.predict(x_test)
    #   fscore = f1_score(y_test, y_pred)
    #   print("F1 score: ", fscore)
      listvar.append(fscore)

  return listvar, max(listvar), sum(listvar) / len(listvar), statistics.stdev(listvar)

x = df.drop("Team Number (linked to FRC Events)", axis = 1)
y = df["Team Number (linked to FRC Events)"]

f1_list, f1_best, f1_mean, f1_std = my_classifier(x, y, classifier = "xgboost", n_folds = 2)
print(f1_list)
print(f1_best)
print(f1_mean)
print(f1_std)
