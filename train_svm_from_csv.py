import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('example_hog_dataset.csv')

X = df.iloc[:, :-2].values  # Все столбцы кроме class и image_file
y_raw = df['class'].values

label_map = {'airplane': 0, 'bird': 1, 'helicopter': 2}
y = np.array([label_map[label] for label in y_raw])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

print(classification_report(
    y_test, clf.predict(X_test),
    labels=[0, 1, 2],
    target_names=['airplane', 'bird', 'helicopter']
))

joblib.dump(clf, 'svm_model_from_csv.pkl')
print("Модель обучена и сохранена как svm_model_from_csv.pkl")
