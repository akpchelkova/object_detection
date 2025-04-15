import cv2
import os
import numpy as np
import joblib
from skimage.feature import hog

# Загрузим модель
clf = joblib.load('svm_model_from_csv.pkl')
inv_label_map = {0: 'airplane', 1: 'bird', 2: 'kite'}

# Настройки HOG
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

input_folder = 'cropped'
output_folder = 'result'
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(('.jpg', '.png')):
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        features = hog(img, **hog_params)

        pred_class = clf.predict([features])[0]
        label = inv_label_map[pred_class]
        print(f"{file_name} → {label}")

        # Визуализация
        img_color = cv2.imread(img_path)
        cv2.putText(img_color, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out_path = os.path.join(output_folder, file_name)
        cv2.imwrite(out_path, img_color)
