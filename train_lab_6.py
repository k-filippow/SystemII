import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os


def load_dataset():
    # Загрузка набора данных CIFAR-10
    print("Загрузка набора данных CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Нормализация данных
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Преобразование меток в one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"Размер обучающей выборки: {x_train.shape}")
    print(f"Размер тестовой выборки: {x_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def create_data_augmentation():
    # Создание аугментации данных для улучшения обобщения
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    return data_augmentation


def create_optimized_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    # Оптимизированная архитектура CNN для быстрого обучения
    model = keras.Sequential([
        # Аугментация данных
        create_data_augmentation(),

        # Первый свёрточный блок
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Второй свёрточный блок
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Третий свёрточный блок
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Глобальный пулинг вместо Flatten для уменьшения параметров
        layers.GlobalAveragePooling2D(),

        # Полносвязные слои
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_simple_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    # Упрощённая архитектура для очень быстрого обучения
    model = keras.Sequential([
        # Аугментация данных
        create_data_augmentation(),

        # Базовая архитектура
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_optimized_model(use_simple_model=False):
    # Обучение оптимизированной модели
    # Загрузка данных
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # Выбор модели
    if use_simple_model:
        print("Использование упрощённой модели...")
        model = create_simple_cnn_model()
        model_name = "simple_cnn_model.h5"
        epochs = 15
        batch_size = 256
    else:
        print("Использование оптимизированной модели...")
        model = create_optimized_cnn_model()
        model_name = "optimized_cnn_model.h5"
        epochs = 20
        batch_size = 128

    # Компиляция модели с оптимизированными параметрами
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Вывод архитектуры модели
    print("\nАрхитектура модели:")
    model.summary()

    # Callbacks для оптимизации обучения
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\nНачало обучения...")
    print(f"Эпох: {epochs}")
    print(f"Размер батча: {batch_size}")

    # Обучение модели
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Оценка модели
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nРезультаты обучения:")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"Потери на тестовой выборке: {test_loss:.4f}")

    # Сохранение финальной модели
    model.save(model_name)
    print(f"Модель сохранена в файл '{model_name}'")

    # Визуализация процесса обучения
    plot_training_history(history, use_simple_model)

    return model, history, (x_test, y_test)


def plot_training_history(history, is_simple_model=False):
    # Визуализация процесса обучения
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # График точности
    ax1.plot(history.history['accuracy'], label='Обучающая точность', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Валидационная точность', linewidth=2)
    ax1.set_title('Точность модели')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График потерь
    ax2.plot(history.history['loss'], label='Обучающие потери', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Валидационные потери', linewidth=2)
    ax2.set_title('Потери модели')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    model_type = "Упрощённая" if is_simple_model else "Оптимизированная"
    plt.suptitle(f'{model_type} модель - Процесс обучения', fontsize=16)
    plt.tight_layout()
    plt.savefig('training_history_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()


def quick_demo():
    # Демо-режим с очень быстрым обучением
    print("\n" + "=" * 50)
    print("ЗАПУСК ДЕМО-РЕЖИМА (Сверхбыстрое обучение)")
    print("=" * 50)

    # Загрузка данных
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Супер простая модель для демо
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Обучение демо-модели (5 эпох)...")
    history = model.fit(
        x_train, y_train,
        batch_size=512,
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nДемо-результат: Точность = {test_accuracy:.4f}")

    model.save('demo_cnn_model.h5')
    print("Демо-модель сохранена в 'demo_cnn_model.h5'")

    return model


if __name__ == "__main__":
    print(" ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ СВЁРТОЧНОЙ СЕТИ ")
    print("Выберите режим обучения:")
    print("1. Демо-режим (5 эпох, очень быстро)")
    print("2. Упрощённая модель (15 эпох)")
    print("3. Оптимизированная модель (20 эпох)")

    choice = input("Введите номер (1-3): ").strip()

    if choice == '1':
        model = quick_demo()
    elif choice == '2':
        model, history, test_data = train_optimized_model(use_simple_model=True)
    elif choice == '3':
        model, history, test_data = train_optimized_model(use_simple_model=False)
    else:
        print("Неверный выбор. Запуск демо-режима...")
        model = quick_demo()