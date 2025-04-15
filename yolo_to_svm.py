import os
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from skimage.feature import hog

# Конфигурация путей
INPUT_IMAGES_DIR = 'images/'      # Папка с исходными изображениями
RESULTS_DIR = 'results/'          # Папка с XML-файлами разметки
SVM_INPUT_DIR = 'svm_input/'      # Папка для сохранения CSV
OUTPUT_CSV = os.path.join(SVM_INPUT_DIR, 'hog_features.csv')

### САША ОТРЕДАЧЬ ПОД РЕАЛЬНЫЕ ПУТИ К ПАПКАМ!!!!!!!!!


# Параметры HOG
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'visualize': False
}

# Создаем папку для выходных данных
os.makedirs(SVM_INPUT_DIR, exist_ok=True)


def parse_xml_annotation(xml_path):
    """Парсит XML файл с аннотациями и возвращает список объектов"""
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
    """Извлекает HOG-признаки для области изображения"""
    xmin, ymin, xmax, ymax = bbox
    roi = image[ymin:ymax, xmin:xmax]

    # Пропускаем пустые области
    if roi.size == 0:
        return None

    # Конвертируем в grayscale и ресайзим
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))

    # Вычисляем HOG-признаки
    features = hog(resized, **HOG_PARAMS)
    return features


# Собираем все данные
all_data = []

# Обрабатываем каждый XML файл в папке results
for xml_file in os.listdir(RESULTS_DIR):
    if not xml_file.endswith('.xml'):
        continue

    # Получаем имя соответствующего изображения
    xml_path = os.path.join(RESULTS_DIR, xml_file)
    image_name = ET.parse(xml_path).getroot().find('filename').text
    image_path = os.path.join(INPUT_IMAGES_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"Изображение {image_name} не найдено, пропускаем")
        continue

    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения {image_name}, пропускаем")
        continue

    # Парсим XML с аннотациями
    objects = parse_xml_annotation(xml_path)

    if not objects:
        print(f"Не найдены объекты в файле {xml_file}, пропускаем")
        continue

    # Обрабатываем каждый объект
    for obj in objects:
        features = extract_hog_features(image, obj['bbox'])
        if features is not None:
            # Добавляем признаки, класс и имя изображения
            row = list(features) + [obj['class'], image_name]
            all_data.append(row)

# Сохраняем результаты в CSV
if all_data:
    # Создаем DataFrame
    num_features = len(all_data[0]) - 2  # Минус класс и имя файла
    columns = [f'hog_{i}' for i in range(num_features)] + ['class', 'image_name']
    df = pd.DataFrame(all_data, columns=columns)

    # Сохраняем в CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Успешно сохранено {len(all_data)} объектов в {OUTPUT_CSV}")
    print("Пример данных:")
    print(df.head())
else:
    print("Нет данных для сохранения")