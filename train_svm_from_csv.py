import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Загрузка CSV
df = pd.read_csv('hog_features (3).csv')

# Отделяем признаки и метки
X = df.iloc[:, :-2].values
y_raw = df['class'].values

# Кодируем текстовые метки в числа
label_map = {'airplane': 0, 'bird': 1, 'helicopter': 2}
y = np.array([label_map[label] for label in y_raw])

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Создаём обратную карту
inv_label_map = {v: k for k, v in label_map.items()}

# Определим реально присутствующие классы
present_labels = sorted(np.unique(y_test))
present_names = [inv_label_map[i] for i in present_labels]

# Печатаем отчёт
print(classification_report(
    y_test, clf.predict(X_test),
    labels=present_labels,
    target_names=present_names
))

joblib.dump(clf, 'svm_model_from_csv.pkl')
print("Модель обучена и сохранена как svm_model_from_csv.pkl")
