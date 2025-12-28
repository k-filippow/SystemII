import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ РЕГУЛЯРИЗАЦИИ")
print("ДЛЯ ПРОГНОЗИРОВАНИЯ ЭНЕРГОПОТРЕБЛЕНИЯ")
print("Датасет: household_power_consumption.txt")
print("=" * 70)



# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ


def load_household_power_data(filepath='household_power_consumption.txt', nrows=None):

    # Загрузка данных household_power_consumption.txt

    print("\n1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("-" * 50)
    print(f"Загрузка файла: {filepath}")


    try:
        # Читаем данные с указанием разделителя и обозначения пропусков
        df = pd.read_csv(filepath, sep=';', low_memory=False,
                         na_values=['?', '??', '???', 'NA', ''],
                         nrows=nrows)

        print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
        print(f"\nСтруктура данных:")
        print(df.info())

        print(f"\nПервые 3 строки:")
        print(df.head(3))

        print(f"\nСтатистика по данным:")
        print(df.describe().round(2))

        # Проверка пропусков
        print(f"\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ПО СТОЛБЦАМ:")
        missing_data = df.isnull().sum()
        for col, missing in missing_data.items():
            if missing > 0:
                print(f"  {col}: {missing} пропусков ({missing / len(df) * 100:.2f}%)")

        # Объединяем Date и Time в один столбец datetime
        print("\nСоздание столбца datetime...")
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                        format='%d/%m/%Y %H:%M:%S')

        # Устанавливаем datetime как индекс
        df.set_index('datetime', inplace=True)

        # Удаляем исходные столбцы Date и Time
        df.drop(['Date', 'Time'], axis=1, inplace=True)

        # Проверяем типы данных
        print("\nПреобразование типов данных...")
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"  {col}: преобразован в числовой тип")
                except:
                    print(f"  {col}: не удалось преобразовать")

        # Заполнение пропусков
        print("\nОбработка пропусков...")
        initial_missing = df.isnull().sum().sum()
        if initial_missing > 0:
            # Для временных рядов используем интерполяцию
            df_interpolated = df.interpolate(method='time', limit_direction='both')

            # Если остались пропуски заполняем средним
            df_filled = df_interpolated.fillna(df_interpolated.mean())

            final_missing = df_filled.isnull().sum().sum()
            print(f"  Пропусков до обработки: {initial_missing}")
            print(f"  Пропусков после обработки: {final_missing}")
            df = df_filled

        # Извлечение временных признаков
        print("\nИзвлечение временных признаков...")
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['month'] = df.index.month
        df['day_of_month'] = df.index.day
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        # Создание циклических признаков для времени
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Создание лаговых признаков (значения в предыдущий час)
        print("Создание лаговых признаков...")
        target_col = 'Global_active_power'
        for lag in [1, 2, 3, 24, 168]:  # 1ч, 2ч, 3ч, 24ч (день), 168ч (неделя)
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        # Удаляем строки с NaN после создания лагов
        df.dropna(inplace=True)

        print(f"\nИтоговый размер данных: {df.shape}")
        print(f"Количество признаков: {df.shape[1]}")

        return df

    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_features_target(df, target_column='Global_active_power'):

    # Подготовка признаков и целевой переменной

    print(f"\nПодготовка данных. Целевая переменная: {target_column}")

    # Целевая переменная
    y = df[target_column]

    # Признаки - все остальные столбцы, кроме целевой
    X = df.drop(columns=[target_column])

    # Проверяем корреляцию признаков с целевой переменной
    print("\nТоп-10 признаков по корреляции с целевой переменной:")
    correlations = df.corr()[target_column].abs().sort_values(ascending=False)
    print(correlations.head(11))  # 10 + сама целевая

    print(f"\nРазмерность данных:")
    print(f"  Признаки (X): {X.shape}")
    print(f"  Целевая (y): {y.shape}")

    return X, y



# 2. МОДЕЛИРОВАНИЕ И РЕГУЛЯРИЗАЦИЯ


def train_and_compare_models(X, y, test_size=0.2, random_state=42):

    # Обучение и сравнение моделей с различной регуляризацией

    print("\n\n2. МОДЕЛИРОВАНИЕ И РЕГУЛЯРИЗАЦИЯ")
    print("-" * 50)

    from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline

    # Для временных рядов используем специальное разделение
    print("Разделение данных на обучающую и тестовую выборки...")

    # Определяем индекс разделения
    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Обучающая выборка: {X_train.shape[0]} записей")
    print(f"Тестовая выборка: {X_test.shape[0]} записей")
    print(f"Процент тестовых данных: {test_size * 100:.1f}%")

    # Масштабирование признаков
    print("\nМасштабирование признаков...")
    # RobustScaler более устойчив к выбросам
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Создание пайплайнов для каждой модели
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=random_state, max_iter=10000),
        'Lasso Regression': Lasso(random_state=random_state, max_iter=10000),
        'ElasticNet': ElasticNet(random_state=random_state, max_iter=10000)
    }

    # Параметры для GridSearchCV
    param_grids = {
        'Ridge Regression': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        'Lasso Regression': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]},
        'ElasticNet': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    }

    # Используем кросс-валидацию для временных рядов
    tscv = TimeSeriesSplit(n_splits=5)

    results = {}
    best_models = {}

    # Обучение и оценка моделей
    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"МОДЕЛЬ: {name}")
        print('=' * 60)

        if name in param_grids:
            # Используем GridSearchCV для подбора гиперпараметров
            print("Подбор гиперпараметров...")
            grid_search = GridSearchCV(
                model,
                param_grids[name],
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Лучшие параметры: {best_params}")
            print(f"Лучший MSE (CV): {-grid_search.best_score_:.4f}")
        else:
            # Линейная регрессия без регуляризации
            best_model = model
            best_model.fit(X_train_scaled, y_train)
            best_params = "Нет (без регуляризации)"

        # Предсказания
        y_pred_train = best_model.predict(X_train_scaled)
        y_pred_test = best_model.predict(X_test_scaled)

        # Метрики
        metrics = {
            'Параметры': best_params,
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Train MAE': mean_absolute_error(y_train, y_pred_train),
            'Test MAE': mean_absolute_error(y_test, y_pred_test),
            'Train R²': r2_score(y_train, y_pred_train),
            'Test R²': r2_score(y_test, y_pred_test),
            'Разница R²': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
            'MAPE (%)': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        }

        # Для линейных моделей получаем коэффициенты
        if hasattr(best_model, 'coef_'):
            metrics['Коэффициенты'] = best_model.coef_
            non_zero = np.sum(np.abs(best_model.coef_) > 1e-10)
            total = len(best_model.coef_)
            metrics['Ненулевых коэффициентов'] = non_zero
            metrics['% ненулевых'] = (non_zero / total * 100)
            metrics['Интерсепт'] = best_model.intercept_

        results[name] = metrics
        best_models[name] = best_model

        print(f"Результаты на тестовой выборке:")
        print(f"  RMSE: {metrics['Test RMSE']:.4f}")
        print(f"  MAE: {metrics['Test MAE']:.4f}")
        print(f"  R²: {metrics['Test R²']:.4f}")
        print(f"  MAPE: {metrics['MAPE (%)']:.2f}%")
        if 'Ненулевых коэффициентов' in metrics:
            print(f"  Ненулевых коэффициентов: {metrics['Ненулевых коэффициентов']}/{len(metrics['Коэффициенты'])}")

    return results, best_models, X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns



# 3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ


def visualize_results(df, results, best_models, X_test_scaled, y_test, scaler, feature_names):

    # Визуализация результатов сравнения моделей

    print("\n\n3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("-" * 50)

    # Создаем папку для результатов если её нет
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    # 3.1. Анализ временного ряда
    print("\nВизуализация временного ряда...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Временной ряд целевой переменной
    sample_data = df['Global_active_power'].iloc[:1000]  # Первые 1000 точек для наглядности
    axes[0, 0].plot(sample_data.index, sample_data.values, linewidth=1)
    axes[0, 0].set_title('Global Active Power (первые 1000 записей)')
    axes[0, 0].set_xlabel('Дата и время')
    axes[0, 0].set_ylabel('Киловатты')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Распределение целевой переменной
    axes[0, 1].hist(df['Global_active_power'].values, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Распределение Global Active Power')
    axes[0, 1].set_xlabel('Киловатты')
    axes[0, 1].set_ylabel('Частота')

    # Потребление по часам
    hourly_avg = df.groupby('hour')['Global_active_power'].mean()
    axes[1, 0].bar(hourly_avg.index, hourly_avg.values, alpha=0.7)
    axes[1, 0].set_title('Среднее потребление по часам суток')
    axes[1, 0].set_xlabel('Час')
    axes[1, 0].set_ylabel('Средние киловатты')
    axes[1, 0].set_xticks(range(0, 24, 2))

    # Потребление по дням недели
    day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    daily_avg = df.groupby('day_of_week')['Global_active_power'].mean()
    axes[1, 1].bar(range(7), daily_avg.values, alpha=0.7, tick_label=day_names)
    axes[1, 1].set_title('Среднее потребление по дням недели')
    axes[1, 1].set_xlabel('День недели')
    axes[1, 1].set_ylabel('Средние киловатты')

    # Корреляционная матрица (только основные признаки)
    main_features = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                     'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    corr_matrix = df[main_features].corr()
    im = axes[2, 0].imshow(corr_matrix.values, cmap='coolwarm', aspect='auto')
    axes[2, 0].set_title('Корреляционная матрица')
    axes[2, 0].set_xticks(range(len(main_features)))
    axes[2, 0].set_yticks(range(len(main_features)))
    axes[2, 0].set_xticklabels(main_features, rotation=45, ha='right')
    axes[2, 0].set_yticklabels(main_features)
    plt.colorbar(im, ax=axes[2, 0])

    # Потребление по месяцам
    monthly_avg = df.groupby('month')['Global_active_power'].mean()
    axes[2, 1].bar(monthly_avg.index, monthly_avg.values, alpha=0.7)
    axes[2, 1].set_title('Среднее потребление по месяцам')
    axes[2, 1].set_xlabel('Месяц')
    axes[2, 1].set_ylabel('Средние киловатты')

    # Устанавливаем подписи месяцев
    month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                   'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    axes[2, 1].set_xticks(range(1, 13))
    axes[2, 1].set_xticklabels(month_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('results/time_series_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3.2. Сравнение метрик моделей
    print("\nСравнение метрик моделей...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Подготовка данных для визуализации
    metrics_df = pd.DataFrame(results).T

    # Тестовый RMSE
    ax = axes[0, 0]
    rmse_values = metrics_df['Test RMSE']
    colors = ['blue', 'green', 'red', 'purple']
    bars = ax.bar(rmse_values.index, rmse_values.values, color=colors)
    ax.set_ylabel('RMSE')
    ax.set_title('Сравнение RMSE на тестовой выборке')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', rotation=45)

    # Тестовый R²
    ax = axes[0, 1]
    r2_values = metrics_df['Test R²']
    bars = ax.bar(r2_values.index, r2_values.values, color=colors)
    ax.set_ylabel('R²')
    ax.set_title('Сравнение R² на тестовой выборке')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', rotation=45)

    # MAPE
    ax = axes[1, 0]
    mape_values = metrics_df['MAPE (%)']
    bars = ax.bar(mape_values.index, mape_values.values, color=colors)
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Средняя абсолютная процентная ошибка (MAPE)')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', rotation=45)

    # % ненулевых коэффициентов
    ax = axes[1, 1]
    if '% ненулевых' in metrics_df.columns:
        nonzero_pct = metrics_df['% ненулевых']
        bars = ax.bar(nonzero_pct.index, nonzero_pct.values, color=colors)
        ax.set_ylabel('% ненулевых коэффициентов')
        ax.set_title('Процент ненулевых коэффициентов')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Данные о коэффициентах\nнедоступны',
                ha='center', va='center', transform=ax.transAxes)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('results/model_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3.3. Важность признаков
    print("\nАнализ важности признаков...")

    # Создаем датафрейм для сравнения коэффициентов
    coef_data = []
    for name, model in best_models.items():
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
                coef_data.append({
                    'Model': name,
                    'Feature': feature,
                    'Coefficient': coef,
                    'Abs_Coefficient': abs(coef)
                })

    if coef_data:
        coef_df = pd.DataFrame(coef_data)

        # Топ-15 признаков по важности для каждой модели
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, model_name in enumerate(best_models.keys()):
            if model_name in coef_df['Model'].unique():
                model_coef = coef_df[coef_df['Model'] == model_name]
                top_features = model_coef.nlargest(10, 'Abs_Coefficient')

                colors = ['red' if coef < 0 else 'green' for coef in top_features['Coefficient']]
                axes[idx].barh(top_features['Feature'], top_features['Coefficient'], color=colors)
                axes[idx].set_xlabel('Значение коэффициента')
                axes[idx].set_title(f'{model_name}\nТоп-10 признаков')
                axes[idx].axvline(x=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()

    # 3.4. Фактические vs Предсказанные значения для лучшей модели
    print("\nВизуализация предсказаний лучшей модели...")

    # Определяем лучшую модель по тестовому R²
    best_model_name = max(results.keys(), key=lambda x: results[x]['Test R²'])
    best_model = best_models[best_model_name]

    # Предсказания на тестовой выборке
    y_pred_test = best_model.predict(X_test_scaled)

    # Выбираем часть тестовых данных для визуализации (первые 200 точек)
    plot_points = min(1000, len(y_test))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # График рассеяния
    axes[0, 0].scatter(y_test[:plot_points], y_pred_test[:plot_points],
                       alpha=0.6, color='blue', s=20)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', lw=2, label='Идеальная предсказательная способность')
    axes[0, 0].set_xlabel('Фактические значения (кВт)')
    axes[0, 0].set_ylabel('Предсказанные значения (кВт)')
    axes[0, 0].set_title(f'Фактические vs Предсказанные значения\n({best_model_name})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Временной ряд предсказаний (первые 100 точек)
    axes[0, 1].plot(range(plot_points), y_test.values[:plot_points],
                    'b-', label='Фактические', alpha=0.7, linewidth=1)
    axes[0, 1].plot(range(plot_points), y_pred_test[:plot_points],
                    'r-', label='Предсказанные', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Временной индекс')
    axes[0, 1].set_ylabel('Global Active Power (кВт)')
    axes[0, 1].set_title(f'Сравнение временных рядов\n({best_model_name})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Ошибки предсказания
    errors = y_pred_test - y_test
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нет ошибки')
    axes[1, 0].axvline(x=errors.mean(), color='green', linestyle='-', linewidth=2,
                       label=f'Средняя ошибка: {errors.mean():.4f}')
    axes[1, 0].set_xlabel('Ошибка предсказания (кВт)')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title('Распределение ошибок предсказания')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Кумулятивное распределение ошибок
    sorted_errors = np.sort(np.abs(errors))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 1].plot(sorted_errors, cumulative, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Абсолютная ошибка (кВт)')
    axes[1, 1].set_ylabel('Доля прогнозов')
    axes[1, 1].set_title('Кумулятивное распределение абсолютных ошибок')
    axes[1, 1].grid(True, alpha=0.3)

    # Добавляем аннотации для порогов ошибок
    for threshold in [0.1, 0.5, 1.0]:
        if threshold < sorted_errors.max():
            idx = np.searchsorted(sorted_errors, threshold)
            if idx < len(cumulative):
                axes[1, 1].axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
                axes[1, 1].text(threshold, cumulative[idx] + 0.02,
                                f'{cumulative[idx] * 100:.1f}%',
                                fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig('results/predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    return best_model_name



# 4. ВЫВОДЫ И ЗАКЛЮЧЕНИЯ


def print_conclusions(results, best_model_name):

    # Вывод заключительных результатов и выводов

    print("\n\n4. ВЫВОДЫ И ЗАКЛЮЧЕНИЕ")
    print("=" * 70)

    # Создаем DataFrame для удобного отображения
    df_results = pd.DataFrame(results).T

    print("\nСВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
    print("-" * 70)

    # Определяем доступные столбцы
    available_cols = []
    for col in ['Test RMSE', 'Test MAE', 'Test R²', 'MAPE (%)', 'Разница R²']:
        if col in df_results.columns:
            available_cols.append(col)

    if available_cols:
        display_df = df_results[available_cols].copy()

        # Форматируем вывод
        def format_value(x):
            if isinstance(x, (int, np.integer)):
                return f"{x}"
            elif isinstance(x, (float, np.floating)):
                if abs(x) < 1000:
                    return f"{x:.4f}"
                else:
                    return f"{x:.2f}"
            else:
                return str(x)

        # Применяем форматирование к каждому столбцу
        formatted_df = display_df.copy()
        for col in formatted_df.columns:
            if col not in ['Параметры', 'Коэффициенты']:
                formatted_df[col] = formatted_df[col].apply(format_value)

        print(formatted_df)
    else:
        print("Нет доступных метрик для отображения")

    print("\n" + "=" * 70)
    print("КЛЮЧЕВЫЕ ВЫВОДЫ:")
    print("=" * 70)

    # Анализ переобучения
    print("\n1. АНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
    for model_name, metrics in results.items():
        if 'Разница R²' in metrics:
            try:
                diff = float(metrics['Разница R²'])
                if diff > 0.15:
                    print(f"   • {model_name}: Сильное переобучение (разница R² = {diff:.4f})")
                elif diff > 0.08:
                    print(f"   • {model_name}: Умеренное переобучение (разница R² = {diff:.4f})")
                elif diff > 0:
                    print(f"   • {model_name}: Легкое переобучение (разница R² = {diff:.4f})")
                else:
                    print(f"   • {model_name}: Хорошая обобщающая способность (разница R² = {diff:.4f})")
            except (ValueError, TypeError):
                print(f"   • {model_name}: Невозможно оценить переобучение")

    # Сравнение моделей
    print("\n2. РЕЙТИНГ МОДЕЛЕЙ ПО ТОЧНОСТИ (Test R²):")
    try:
        # Фильтруем модели с доступным Test R²
        valid_models = []
        for name, metrics in results.items():
            if 'Test R²' in metrics and metrics['Test R²'] is not None:
                try:
                    r2 = float(metrics['Test R²'])
                    valid_models.append((name, r2, metrics.get('Test RMSE', 0)))
                except (ValueError, TypeError):
                    continue

        if valid_models:
            sorted_models = sorted(valid_models, key=lambda x: x[1], reverse=True)
            for i, (model_name, r2, rmse) in enumerate(sorted_models):
                print(f"   {i + 1}. {model_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        else:
            print("   Нет данных для сравнения")
    except Exception as e:
        print(f"   Ошибка при сравнении моделей: {e}")

    # Анализ коэффициентов
    print("\n3. АНАЛИЗ КОЭФФИЦИЕНТОВ И ОТБОРА ПРИЗНАКОВ:")
    for model_name, metrics in results.items():
        if 'Ненулевых коэффициентов' in metrics:
            try:
                n_coef = int(metrics['Ненулевых коэффициентов'])
                total_coef = len(metrics['Коэффициенты']) if 'Коэффициенты' in metrics else 'N/A'
                if total_coef != 'N/A':
                    total_coef = int(total_coef)
                    pct = (n_coef / total_coef * 100) if total_coef > 0 else 0
                    print(f"   • {model_name}: {n_coef} ненулевых коэффициентов из {total_coef} ({pct:.1f}%)")
                else:
                    print(f"   • {model_name}: {n_coef} ненулевых коэффициентов")
            except (ValueError, TypeError):
                print(f"   • {model_name}: Невозможно проанализировать коэффициенты")

    # Определяем тип данных по метрикам
    print("\n4. ХАРАКТЕРИСТИКА ДАННЫХ И МОДЕЛЕЙ:")
    if best_model_name in results:
        best_metrics = results[best_model_name]

        if 'Test R²' in best_metrics and best_metrics['Test R²'] is not None:
            try:
                r2 = float(best_metrics['Test R²'])
                if r2 > 0.8:
                    data_quality = "Отличное"
                elif r2 > 0.6:
                    data_quality = "Хорошее"
                elif r2 > 0.4:
                    data_quality = "Удовлетворительное"
                else:
                    data_quality = "Низкое"

                print(f"   Качество прогнозирования: {data_quality} (R² = {r2:.4f})")
            except (ValueError, TypeError):
                print(f"   Качество прогнозирования: Невозможно оценить")

        if 'Test MAE' in best_metrics:
            try:
                mae = float(best_metrics['Test MAE'])
                print(f"   Средняя абсолютная ошибка: {mae:.4f} кВт")
            except (ValueError, TypeError):
                pass

        if 'MAPE (%)' in best_metrics:
            try:
                mape = float(best_metrics['MAPE (%)'])
                print(f"   Средняя процентная ошибка: {mape:.2f}%")
            except (ValueError, TypeError):
                pass
    else:
        print("   Лучшая модель не найдена в результатах")

    print("\n5. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
    print(f"   Лучшая модель: {best_model_name}")

    if "Lasso" in best_model_name:
        print("   Lasso эффективно отсеял неважные признаки, упростив модель")
        print("   Рекомендуется для интерпретации наиболее значимых факторов")
    elif "Ridge" in best_model_name:
        print("   Ridge показал хорошую устойчивость к мультиколлинеарности")
        print("   Рекомендуется при наличии коррелированных предикторов")
    elif "ElasticNet" in best_model_name:
        print("   ElasticNet нашел баланс между отбором признаков и устойчивостью")
        print("   Хороший выбор для компромисса между точностью и интерпретируемостью")
    else:
        print("   Линейная регрессия без регуляризации может быть достаточно хороша")
        print("   Однако регуляризация обычно улучшает обобщающую способность")

    print("\n" + "=" * 70)
    print(f"ИТОГ: {best_model_name} показал наилучшие результаты")
    print("для прогнозирования энергопотребления на данном наборе данных.")
    print("=" * 70)


def save_detailed_results(results, best_model_name, filename='results/detailed_analysis.txt'):

    # Сохранение подробных результатов в файл

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ДЕТАЛЬНЫЙ ОТЧЕТ ПО СРАВНИТЕЛЬНОМУ АНАЛИЗУ МЕТОДОВ РЕГУЛЯРИЗАЦИИ\n")
        f.write("=" * 80 + "\n\n")

        f.write("ДАТАСЕТ: household_power_consumption.txt\n")
        f.write("ЦЕЛЕВАЯ ПЕРЕМЕННАЯ: Global_active_power\n")
        f.write(f"ВРЕМЯ АНАЛИЗА: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("РЕЗУЛЬТАТЫ МОДЕЛЕЙ:\n")
        f.write("-" * 80 + "\n")

        for model_name, metrics in results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  Параметры модели: {metrics['Параметры']}\n")
            f.write(f"  Метрики на обучающей выборке:\n")
            f.write(f"    RMSE: {metrics['Train RMSE']:.6f}\n")
            f.write(f"    MAE: {metrics['Train MAE']:.6f}\n")
            f.write(f"    R²: {metrics['Train R²']:.6f}\n")

            f.write(f"  Метрики на тестовой выборке:\n")
            f.write(f"    RMSE: {metrics['Test RMSE']:.6f}\n")
            f.write(f"    MAE: {metrics['Test MAE']:.6f}\n")
            f.write(f"    R²: {metrics['Test R²']:.6f}\n")
            f.write(f"    MAPE: {metrics.get('MAPE (%)', 'N/A')}\n")
            f.write(f"    Разница R² (train-test): {metrics.get('Разница R²', 'N/A')}\n")

            if 'Ненулевых коэффициентов' in metrics:
                f.write(f"  Статистика коэффициентов:\n")
                f.write(f"    Всего коэффициентов: {len(metrics['Коэффициенты'])}\n")
                f.write(f"    Ненулевых коэффициентов: {metrics['Ненулевых коэффициентов']}\n")
                f.write(f"    Процент ненулевых: {metrics.get('% ненулевых', 'N/A')}\n")
                f.write(f"    Интерсепт: {metrics.get('Интерсепт', 'N/A')}\n")

                if 'Коэффициенты' in metrics and metrics['Коэффициенты'] is not None:
                    f.write(f"  Топ-10 признаков по абсолютному значению коэффициента:\n")
                    try:
                        # Создаем DataFrame с коэффициентами
                        coef_list = []
                        for idx, coef in enumerate(metrics['Коэффициенты']):
                            coef_list.append({
                                'Признак': idx,
                                'Коэффициент': float(coef),
                                'Абс_значение': abs(float(coef))
                            })

                        coef_df = pd.DataFrame(coef_list)
                        coef_df = coef_df.sort_values('Абс_значение', ascending=False)

                        for i in range(min(10, len(coef_df))):
                            row = coef_df.iloc[i]
                            # Безопасное форматирование
                            feature_idx = int(row['Признак'])
                            coef_value = float(row['Коэффициент'])
                            f.write(f"    {i + 1:2d}. Признак {feature_idx:3d}: {coef_value:10.6f}\n")
                    except Exception as e:
                        f.write(f"    Ошибка при форматировании коэффициентов: {e}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ВЫВОДЫ И РЕКОМЕНДАЦИИ:\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name}\n")
        if best_model_name in results:
            best_metrics = results[best_model_name]
            f.write(f"  Test R²: {best_metrics.get('Test R²', 'N/A')}\n")
            f.write(f"  Test RMSE: {best_metrics.get('Test RMSE', 'N/A')}\n")
            f.write(f"  Test MAE: {best_metrics.get('Test MAE', 'N/A')}\n")
            f.write(f"  MAPE: {best_metrics.get('MAPE (%)', 'N/A')}\n\n")

        f.write("ОБЩИЕ ВЫВОДЫ:\n")
        f.write("1. Регуляризация улучшает обобщающую способность моделей\n")
        f.write("2. Lasso эффективен для отбора наиболее значимых признаков\n")
        f.write("3. Ridge обеспечивает устойчивость к мультиколлинеарности\n")
        f.write("4. ElasticNet предлагает сбалансированный подход\n")
        f.write("5. Для прогнозирования энергопотребления важны временные признаки\n")



# ОСНОВНАЯ ПРОГРАММА


def main():

    # Основная функция выполнения анализа

    print("\n" + "=" * 70)
    print("НАЧАЛО АНАЛИЗА ДАТАСЕТА ПО ЭНЕРГОПОТРЕБЛЕНИЮ")
    print("=" * 70)

    # Параметры загрузки
    DATA_FILE = 'household_power_consumption.txt'
    SAMPLE_SIZE = 500000  # Установите None для загрузки всех данных

    # Загрузка данных
    print(f"\nЗагружаем данные из файла: {DATA_FILE}")
    if SAMPLE_SIZE:
        print(f"Загружаем выборку: {SAMPLE_SIZE} записей")

    df = load_household_power_data(DATA_FILE, nrows=SAMPLE_SIZE)

    if df is None:
        print("Не удалось загрузить данные. Проверьте путь к файлу.")
        return

    # Подготовка признаков и целевой переменной
    X, y = prepare_features_target(df, target_column='Global_active_power')

    # Обучение и сравнение моделей
    results, best_models, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = train_and_compare_models(
        X, y, test_size=0.2, random_state=42
    )

    # Визуализация результатов
    best_model_name = visualize_results(df, results, best_models, X_test_scaled,
                                        y_test, scaler, feature_names)

    # Вывод заключений
    print_conclusions(results, best_model_name)

    # Сохранение детальных результатов
    save_detailed_results(results, best_model_name)

    print("\n" + "=" * 70)
    print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
    print("=" * 70)
    print("\nСОЗДАННЫЕ ФАЙЛЫ:")
    print("  results/time_series_analysis.png      - Анализ временного ряда")
    print("  results/model_metrics_comparison.png  - Сравнение метрик моделей")
    print("  results/feature_importance.png        - Важность признаков")
    print("  results/predictions_visualization.png - Визуализация предсказаний")
    print("  results/detailed_analysis.txt         - Детальный отчет")
    print("\nРекомендуется использовать модель:", best_model_name)
    print("=" * 70)



# ЗАПУСК ПРОГРАММЫ


if __name__ == "__main__":
    # Запускаем основной анализ
    main()