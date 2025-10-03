import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

# Загрузка файла
FILE_NAME = "household_power_consumption.txt"

df = pd.read_csv(
    FILE_NAME,
    sep=';',
    na_values=['?'],
    low_memory=False
)

# создаём datetime из двух колонок
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df = df.drop(columns=['Date', 'Time'])

# Предобработка
num_cols = [
    'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# удалим пустые строки по целевой переменной
df = df.dropna(subset=['Global_active_power']).copy()
df = df.set_index('datetime').sort_index()

print("Размер исходных данных:", df.shape)

# Агрегация
# усреднение по часам
df = df[num_cols].resample('h').mean().dropna()

print("Размер данных после агрегации по часам:", df.shape)

# Признаки и цель
y = df['Global_active_power'].values

X_num = df[['Global_reactive_power','Voltage','Global_intensity',
            'Sub_metering_1','Sub_metering_2','Sub_metering_3']].values

hours = df.index.hour.values.reshape(-1,1)
dow = df.index.weekday.values.reshape(-1,1)

X = np.hstack([X_num, hours, dow])

feature_names = ['Реактивная мощность','Напряжение','Сила тока',
                 'Субсчетчик 1','Субсчетчик 2','Субсчетчик 3','Час','День недели']

# Разбиение
def custom_train_test_split(X, y, train_frac=0.8, random_state=42):
    n = X.shape[0]
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    n_train = int(np.floor(train_frac * n))
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = custom_train_test_split(X, y, train_frac=0.8, random_state=1)

print("Размер обучающей выборки:", X_train.shape, "| Размер тестовой выборки:", X_test.shape)

# Линейная регрессия
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_train_pred = lr.predict(X_train_scaled)
y_test_pred = lr.predict(X_test_scaled)

print("\nЛинейная регрессия")
print(f"R² (обучение): {r2_score(y_train, y_train_pred):.4f}, R² (тест): {r2_score(y_test, y_test_pred):.4f}")
print(f"MSE (обучение): {mean_squared_error(y_train, y_train_pred):.4f}, MSE (тест): {mean_squared_error(y_test, y_test_pred):.4f}")

# Полиномиальные модели
degrees = range(1, 6)
r2_train_poly, r2_test_poly = [], []

print("\nПолиномиальные модели ")
for d in degrees:
    model = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                          StandardScaler(),
                          LinearRegression())
    model.fit(X_train, y_train)
    r2_train_poly.append(r2_score(y_train, model.predict(X_train)))
    r2_test_poly.append(r2_score(y_test, model.predict(X_test)))
    print(f"Степень={d}: R² (обучение)={r2_train_poly[-1]:.4f}, R² (тест)={r2_test_poly[-1]:.4f}")

plt.figure(figsize=(7,5))
plt.plot(degrees, r2_train_poly, marker='o', label='R² (обучение)')
plt.plot(degrees, r2_test_poly, marker='o', label='R² (тест)')
plt.xlabel("Степень полинома")
plt.ylabel("R²")
plt.title("Зависимость точности от степени полинома")
plt.legend()
plt.grid()
plt.show()

# Регуляризация
best_degree = degrees[np.argmax(r2_test_poly)]  # выбираем степень с лучшим тестовым R^2
alphas = np.logspace(-4, 4, 9)

r2_train_ridge, r2_test_ridge = [], []

print(f"\nRidge-регрессия (степень полинома={best_degree}) ")
for a in alphas:
    model = make_pipeline(PolynomialFeatures(degree=best_degree, include_bias=False),
                          StandardScaler(),
                          Ridge(alpha=a))
    model.fit(X_train, y_train)
    r2_train_ridge.append(r2_score(y_train, model.predict(X_train)))
    r2_test_ridge.append(r2_score(y_test, model.predict(X_test)))
    print(f"alpha={a:.1e}: R² (обучение)={r2_train_ridge[-1]:.4f}, R² (тест)={r2_test_ridge[-1]:.4f}")

plt.figure(figsize=(7,5))
plt.semilogx(alphas, r2_train_ridge, marker='o', label='R² (обучение)')
plt.semilogx(alphas, r2_test_ridge, marker='o', label='R² (тест)')
plt.xlabel("Alpha (Ridge)")
plt.ylabel("R²")
plt.title(f"Ridge-регрессия, степень полинома={best_degree}")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Lasso-регрессия
print(f"\n Lasso-регрессия (степень полинома={best_degree}) ")
r2_test_lasso = []
for a in alphas:
    model = make_pipeline(PolynomialFeatures(degree=best_degree, include_bias=False),
                          StandardScaler(),
                          Lasso(alpha=a, max_iter=5000))
    model.fit(X_train, y_train)
    r2_test_lasso.append(r2_score(y_test, model.predict(X_test)))
    print(f"alpha={a:.1e}: R² (тест)={r2_test_lasso[-1]:.4f}")

plt.figure(figsize=(7,5))
plt.semilogx(alphas, r2_test_lasso, marker='o', color='purple')
plt.xlabel("Alpha (Lasso)")
plt.ylabel("R² (тест)")
plt.title(f"Lasso-регрессия, степень полинома={best_degree}")
plt.grid(True, which="both", ls="--")
plt.show()
