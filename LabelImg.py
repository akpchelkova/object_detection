import os
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from skimage.feature import hog

# Пути к папкам
images_dir = 'dataset/bird/images/'
annotations_dir = 'dataset/bird/annotations/'
output_csv = 'hog_features.csv'

# Параметры HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)


def parse_xml(xml_path):
    """Извлекает все объекты из XML файла с их координатами и классами"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'class': class_name,
            'bbox': (xmin, ymin, xmax, ymax)
        })

    return objects


def extract_hog_features(image, bbox):
    """Вычисляет HOG-признаки для области изображения"""
    xmin, ymin, xmax, ymax = bbox
    roi = image[ymin:ymax, xmin:xmax]

    # Если область пустая, пропускаем
    if roi.size == 0:
        return None

    # Преобразуем в grayscale и изменяем размер для стандартизации
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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
for image_file in os.listdir(images_dir):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Получаем соответствующий XML файл
    xml_file = os.path.splitext(image_file)[0] + '.xml'
    xml_path = os.path.join(annotations_dir, xml_file)

    if not os.path.exists(xml_path):
        print(f"Не найден XML файл для изображения: {image_file}")
        continue

    # Загружаем изображение
    image_path = os.path.join(images_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_file}")
        continue

    # Парсим XML
    objects = parse_xml(xml_path)
    if not objects:
        print(f"Не найдены объекты в файле: {xml_file}")
        continue

    # Обрабатываем каждый объект
    for obj in objects:
        # Извлекаем HOG-признаки для объекта
        features = extract_hog_features(image, obj['bbox'])
        if features is not None:
            # Добавляем признаки, метку класса и название файла
            row = list(features) + [obj['class'], image_file]
            data.append(row)

# Создаем DataFrame
if data:
    # Создаем заголовки для признаков
    num_features = len(data[0]) - 2  # -2 потому что последние два элемента - класс и имя файла
    feature_columns = [f'hog_{i}' for i in range(num_features)]
    columns = feature_columns + ['class', 'image_file']

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Данные сохранены в {output_csv}")
    print(f"Обработано объектов: {len(data)}")
    print(f"Размерность признаков: {num_features}")
    print(f"Пример данных:\n{df.head()}")
else:
    print("Нет данных для сохранения")