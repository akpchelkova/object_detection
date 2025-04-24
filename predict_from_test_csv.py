import pandas as pd
import joblib
import os
import cv2

# === 1. Загрузка обученной модели
clf = joblib.load('svm_model_from_csv.pkl')
inv_label_map = {0: 'airplane', 1: 'bird', 2: 'helicopter'}

# === 2. Загрузка HOG-признаков
df = pd.read_csv('test (3).csv')
X = df.iloc[:, :-2].values  # Берём только признаки

# === 3. Получаем список файлов в 'cropped/' по порядку
input_folder = 'cropped'
output_folder = 'result'
os.makedirs(output_folder, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.jpg', '.png'))
])

# === 4. Проверка на совпадение кол-ва файлов и признаков
if len(image_files) != len(X):
    print(f"Количество изображений ({len(image_files)}) не совпадает с количеством HOG-векторов ({len(X)})")
    exit()

# === 5. Предсказание
preds = clf.predict(X)
labels = [inv_label_map[p] for p in preds]

# === 6. Визуализация и сохранение
for filename, label in zip(image_files, labels):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Пропущено: {filename}")
        continue

    cv2.putText(img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    result_path = os.path.join(output_folder, filename)
    cv2.imwrite(result_path, img)

print("Все изображения подписаны и сохранены в result/")
