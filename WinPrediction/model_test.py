

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


      searcher = RandomizedSearchCV(clf, parameters)
      searcher.fit(x_train, y_train)
      print("Best params: ", searcher.best_params_)
      print("Best score: ", searcher.best_score_)
      y_pred = searcher.predict(x_test)
      fscore = f1_score(y_test, y_pred)
      print("F1 score: ", fscore)
      listvar.append(fscore)

  return listvar, max(listvar), sum(listvar) / len(listvar), statistics.stdev(listvar)

x = train.drop("ACTION", axis = 1)
y = train["ACTION"]

f1_list, f1_best, f1_mean, f1_std = my_classifier(x, y, classifier = "xgboost", n_folds = 10)
print(f1_list)
print(f1_best)
print(f1_mean)
print(f1_std)