import os
import cv2
import pandas as pd
from skimage.feature import hog

# Пути к папкам
cropped_dir = 'content/cropped/'  # Папка с обрезанными изображениями от YOLO
output_csv = 'hog_features.csv'

# Параметры HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)


def extract_hog_features(image):
    """Вычисляет HOG-признаки для всего изображения"""
    # Преобразуем в grayscale и изменяем размер для стандартизации
    if len(image.shape) > 2:  # Если цветное изображение
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, (64, 64))  # Стандартный размер для HOG

    # Вычисляем HOG-признаки
    features = hog(resized,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   visualize=False)

    return features


# Собираем данные
data = []
for image_file in os.listdir(cropped_dir):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Загружаем изображение
    image_path = os.path.join(cropped_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_file}")
        continue

    # Извлекаем HOG-признаки для всего изображения
    features = extract_hog_features(image)
    if features is not None:
        # Пример: если имя файла "test_0.jpg", то исходное было "test.jpg"
        original_name = '_'.join(image_file.split('_')[:-1]) + '.jpg'

        # Добавляем признаки и название исходного файла
        row = list(features) + [original_name, image_file]
        data.append(row)

# Создаем DataFrame
if data:
    # Создаем заголовки для признаков
    num_features = len(data[0]) - 2  # -2 потому что последние два элемента - имена файлов
    feature_columns = [f'hog_{i}' for i in range(num_features)]
    columns = feature_columns + ['original_image', 'cropped_image']

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Данные сохранены в {output_csv}")
    print(f"Обработано изображений: {len(data)}")
    print(f"Размерность признаков: {num_features}")
    print(f"Пример данных:\n{df.head()}")
else:
    print("Нет данных для сохранения. Проверьте папку cropped_dir.")