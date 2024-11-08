from scipy.stats import randint
import pandas as pd
import numpy as np
import pydot
from IPython.core.display_functions import display

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn import svm

# Tree visualization
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

df = pd.read_csv('../../data/bank-full.csv', delimiter=';')

df['default'] = df['default'].map({'yes': 1, 'no': 0})
df['y'] = df['y'].map({'yes': 1, 'no': 0})


cat_features = df.select_dtypes(include=['object','bool']).columns.values

df_encoded = pd.get_dummies(df, columns=cat_features)

df_encoded.drop(['poutcome_unknown', 'contact_unknown', 'education_unknown', 'job_unknown'], axis=1, inplace=True)

X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

#print(df_encoded.corr())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(len(rf.estimators_))

# for i in range(3):
#     tree = rf.estimators_[i]
#     dot_data = export_graphviz(
#         tree,
#         feature_names=X.columns,
#         filled=True,
#         max_depth=2,
#         rounded=True,
#         proportion=True
#     )
#     (graph,) = pydot.graph_from_dot_data(dot_data)
#     graph.write_png(f'tree_{i}.png')

# param_dist = {
#     'n_estimators': randint(50, 500),
#     'max_depth': randint(1, 10)
# }
#
# rf = RandomForestClassifier()
#
# rand_search = RandomizedSearchCV(
#     rf,
#     param_distributions=param_dist,
#     n_iter=5,
#     cv=5
# )
#
# rand_search.fit(X_train, y_train)
#
# best_rf = rand_search.best_estimator_
#
# print("Best Estimator: ", rand_search.best_params_)
# print("Accuracy Best estimator: ", accuracy_score(y_test, best_rf.predict(X_test)))
#

#SVM

svm_clf = svm.SVC(kernel='linear')

svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')