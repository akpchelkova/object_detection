import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('example_hog_dataset.csv', header=None)

X = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

label_map = {'airplane': 0, 'bird': 1, 'kite': 2}
y = np.array([label_map[label] for label in y_raw])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

joblib.dump(clf, 'svm_model_from_csv.pkl')
print("Модель сохранена как svm_model_from_csv.pkl")
