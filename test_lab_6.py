import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob


class OptimizedCNNClassifier:
    # Класс для работы с оптимизированной CNN моделью

    def __init__(self, model_path=None):
        # Инициализация классификатора
        self.model = None
        self.class_names = [
            'самолет', 'автомобиль', 'птица', 'кот', 'олень',
            'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик'
        ]

        # Автопоиск модели если путь не указан
        if model_path is None:
            model_path = self.find_model_file()

        self.load_model(model_path)

    def find_model_file(self):
        # Автопоиск файла модели
        possible_files = [
            'optimized_cnn_model.h5',
            'simple_cnn_model.h5',
            'demo_cnn_model.h5',
            'best_model.h5',
            'cnn_model.h5'
        ]

        for file in possible_files:
            if os.path.exists(file):
                print(f"Найдена модель: {file}")
                return file

        print("Файл модели не найден! Запустите сначала обучение.")
        return None

    def load_model(self, model_path):
        # Загрузка модели из файла
        if model_path is None or not os.path.exists(model_path):
            print("Ошибка: Файл модели не существует!")
            return False

        try:
            self.model = keras.models.load_model(model_path)
            print(f" Модель успешно загружена из {model_path}")
            print(f" Архихитектура модели:")
            self.model.summary()
            return True
        except Exception as e:
            print(f" Ошибка при загрузке модели: {e}")
            return False

    def preprocess_image(self, image_path, target_size=(32, 32)):
        # Предобработка изображения для классификации
        try:
            # Загрузка изображения
            image = Image.open(image_path)

            # Конвертация в RGB если необходимо
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Изменение размера
            image = image.resize(target_size)

            # Преобразование в numpy array и нормализация
            image_array = np.array(image).astype('float32') / 255.0

            # Добавление dimension для батча
            image_array = np.expand_dims(image_array, axis=0)

            return image_array
        except Exception as e:
            print(f" Ошибка при обработке изображения: {e}")
            return None

    def predict_image(self, image_path):
        # Предсказание класса для одного изображения
        if self.model is None:
            print(" Модель не загружена!")
            return None

        # Предобработка изображения
        processed_image = self.preprocess_image(image_path)

        if processed_image is None:
            return None

        # Предсказание
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Вывод результата
        self.display_prediction(image_path, predicted_class, confidence, predictions[0])

        return predicted_class, confidence, predictions[0]

    def display_prediction(self, image_path, predicted_class, confidence, all_predictions):
        # Отображение результата предсказания
        plt.figure(figsize=(14, 6))

        # Отображение изображения
        plt.subplot(1, 2, 1)
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f'Предсказанный класс: {self.class_names[predicted_class]}\nУверенность: {confidence:.2%}',
                  fontsize=14, pad=20)
        plt.axis('off')

        # Отображение вероятностей для всех классов
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(self.class_names))
        colors = ['lightgray'] * len(self.class_names)
        colors[predicted_class] = 'lightcoral'

        bars = plt.barh(y_pos, all_predictions, color=colors, alpha=0.7)
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Вероятность', fontsize=12)
        plt.title('Вероятности для всех классов', fontsize=14)
        plt.xlim(0, 1)

        # Добавление значений на бары
        for i, (bar, prob) in enumerate(zip(bars, all_predictions)):
            if prob > 0.1:  # Показываем только значимые вероятности
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{prob:.2%}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('prediction_result_optimized.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Вывод в консоль
        print(f"\n Результат классификации:")
        print(f"   Класс: {self.class_names[predicted_class]}")
        print(f"   Уверенность: {confidence:.2%}")
        print(f"   Топ-3 предсказания:")

        # Сортировка по уверенности
        top_indices = np.argsort(all_predictions)[-3:][::-1]
        for i, idx in enumerate(top_indices):
            prob = all_predictions[idx]
            print(f"   {i + 1}. {self.class_names[idx]}: {prob:.2%}")

    def evaluate_on_test_data(self):
        # Оценка модели на тестовых данных CIFAR-10
        if self.model is None:
            print(" Модель не загружена!")
            return None

        print("\n Загрузка тестовых данных CIFAR-10 для оценки...")
        (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Предобработка данных
        x_test = x_test.astype('float32') / 255.0
        y_test_categorical = keras.utils.to_categorical(y_test, 10)

        # Оценка модели
        print("Выполняется оценка модели...")
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test_categorical, verbose=1)

        print(f"\n Результаты оценки на тестовых данных:")
        print(f"   Потери: {test_loss:.4f}")
        print(f"   Точность: {test_accuracy:.4f} ({test_accuracy:.2%})")

        # Дополнительная метрика - точность по классам
        print("Вычисление точности по классам...")
        predictions = self.model.predict(x_test, verbose=0)  # Теперь переменная определена
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = y_test.flatten()

        # Вычисление точности по классам
        class_accuracy = {}
        for i in range(10):
            mask = actual_classes == i
            if np.sum(mask) > 0:
                accuracy = np.mean(predicted_classes[mask] == i)
                class_accuracy[self.class_names[i]] = accuracy

        print(f"\n Точность по классам:")
        for class_name, acc in class_accuracy.items():
            print(f"   {class_name}: {acc:.2%}")

        return test_accuracy

    def batch_predict(self, images_folder):
        # Пакетная классификация изображений из папки
        if self.model is None:
            print(" Модель не загружена!")
            return None

        if not os.path.exists(images_folder):
            print(f" Папка {images_folder} не существует!")
            return None

        # Поиск изображений
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))

        if not image_files:
            print(" В папке нет изображений!")
            return None

        print(f"\n Найдено {len(image_files)} изображений для классификации...")

        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n Обработка {i}/{len(image_files)}: {os.path.basename(image_file)} ---")
            result = self.predict_image(image_file)
            if result:
                predicted_class, confidence, _ = result
                results.append((image_file, predicted_class, confidence))

        # Сводка результатов
        if results:
            print(f"\n{'=' * 50}")
            print(" СВОДКА РЕЗУЛЬТАТОВ ПАКЕТНОЙ КЛАССИФИКАЦИИ")
            print(f"{'=' * 50}")
            for image_file, pred_class, confidence in results:
                filename = os.path.basename(image_file)
                class_name = self.class_names[pred_class]
                print(f"{filename} →  {class_name} ({confidence:.2%})")

        return results

    def predict_single_image(self, image_array):
        # Предсказание для одного изображения
        if self.model is None:
            print(" Модель не загружена!")
            return None

        # Предсказание
        predictions = self.model.predict(image_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return predicted_class, confidence, predictions[0]


def main():
    # Основная функция программы тестирования
    print("ТЕСТИРОВАНИЕ ОПТИМИЗИРОВАННОЙ CNN ")

    # Инициализация классификатора
    classifier = OptimizedCNNClassifier()

    if classifier.model is None:
        print("Не удалось загрузить модель. Завершение работы.")
        return

    while True:
        print("\n" + "=" * 50)
        print(" МЕНЮ ТЕСТИРОВАНИЯ:")
        print("1. Классифицировать одно изображение")
        print("2. Пакетная классификация изображений из папки")
        print("3. Оценка на тестовых данных CIFAR-10")
        print("4. Информация о модели")
        print("5. Выход")
        print("=" * 50)

        choice = input("Выберите опцию (1-5): ").strip()

        if choice == '1':
            image_path = input("Введите путь к изображению: ").strip()
            if os.path.exists(image_path):
                classifier.predict_image(image_path)
            else:
                print(" Файл не существует!")

        elif choice == '2':
            folder_path = input("Введите путь к папке с изображениями: ").strip()
            if not folder_path:
                folder_path = "test_images"  # Папка по умолчанию
            classifier.batch_predict(folder_path)

        elif choice == '3':
            classifier.evaluate_on_test_data()

        elif choice == '4':
            if classifier.model:
                print("\n ИНФОРМАЦИЯ О МОДЕЛИ:")
                print(f"   Входная форма: {classifier.model.input_shape}")
                print(f"   Выходная форма: {classifier.model.output_shape}")
                print(f"   Количество слоёв: {len(classifier.model.layers)}")
                print(f"   Количество параметров: {classifier.model.count_params():,}")

        elif choice == '5':
            print(" Выход из программы.")
            break

        else:
            print(" Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()